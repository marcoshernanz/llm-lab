"""Phase 5 experiment 007: the decoder with QK-Norm and gated attention."""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset  # pyright: ignore
from torch import nn

DATASET_NAME = "roneneldan/TinyStories"
DATASET_CONFIG = None
TRAIN_SPLIT = "train[:20000]"
VAL_SPLIT = "validation[:2000]"
TEXT_COLUMN = "text"
DEVICE = "mps"
SEED = 1337

# Tensor shapes:
# B: batch size
# T: sequence length
# D: model dim
# V: vocab size
# H: a head count, Hq or Hkv depending on the projection
# Hq: number of query heads
# Hkv: number of key and value heads
# Dh: head dim, D // Hq
# Dff: feed-forward dim

CONTEXT_LEN = 256
D_MODEL = 128
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
assert NUM_Q_HEADS % NUM_KV_HEADS == 0
assert D_MODEL % NUM_Q_HEADS == 0
D_HEAD = D_MODEL // NUM_Q_HEADS
D_FFN = 344
NUM_BLOCKS = 8
INIT_STD = 0.02
ROPE_BASE = 10000.0
NORM_EPS = 1e-5
WINDOW_SIZE = 64
GLOBAL_EVERY = 4
assert NUM_BLOCKS % GLOBAL_EVERY == 0

BATCH_SIZE = 32
LEARNING_RATE = 3e-3
GRAD_CLIP_NORM = 1.0
TRAIN_STEPS = 3_000
EVAL_INTERVAL = 250
EVAL_BATCHES = 32


class RMSNorm(nn.Module):
    """Scale each embedding vector by its root mean square and a learned gain."""

    def __init__(self, dim: int):
        """Create the learned gain parameter for one normalized axis."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = NORM_EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., dim]
        """Return the normalized and scaled input."""
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)  # [..., 1]
        normalized = x * torch.rsqrt(mean_square + self.eps)  # [..., dim]
        return normalized * self.weight  # [..., dim]


class CausalSelfAttention(nn.Module):
    """Apply masked grouped-query self-attention over one sequence."""

    causal_mask: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, is_global: bool):
        """Create the projections, the norms, the causal mask, and the rotation tables."""
        super().__init__()
        self.is_global = is_global

        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, NUM_KV_HEADS * D_HEAD, bias=False)
        self.v_proj = nn.Linear(D_MODEL, NUM_KV_HEADS * D_HEAD, bias=False)
        self.g_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

        self.q_norm = RMSNorm(D_HEAD)
        self.k_norm = RMSNorm(D_HEAD)

        mask = torch.ones(CONTEXT_LEN, CONTEXT_LEN, dtype=torch.bool).triu(1)  # [T, T]
        if not is_global:
            mask |= torch.ones(CONTEXT_LEN, CONTEXT_LEN, dtype=torch.bool).tril(-WINDOW_SIZE)
        self.register_buffer("causal_mask", mask)

        inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, D_HEAD, 2) / D_HEAD))  # [Dh/2]
        positions = torch.arange(CONTEXT_LEN)  # [T]
        angles = positions[:, None] * inv_freq[None, :]  # [T, Dh/2]
        angles = torch.cat([angles, angles], dim=-1)  # [T, Dh]
        self.register_buffer("rope_cos", angles.cos())
        self.register_buffer("rope_sin", angles.sin())

    def split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:  # [B, T, H*Dh]
        """Split the projection into separate attention heads."""
        batch_size, seq_len, _ = x.size()
        x = x.reshape(batch_size, seq_len, num_heads, D_HEAD)  # [B, T, H, Dh]
        return x.transpose(1, 2)  # [B, H, T, Dh]

    def repeat_kv_heads(self, x: torch.Tensor) -> torch.Tensor:  # [B, Hkv, T, Dh]
        """Share each key or value head across its group of query heads."""
        return x.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)  # [B, Hq, T, Dh]

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:  # [B, Hq, T, Dh]
        """Merge the attention heads back into one embedding axis."""
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # [B, T, Hq, Dh]
        return x.reshape(batch_size, seq_len, D_MODEL)  # [B, T, D]

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:  # [B, H, T, Dh]
        """Pair each feature with the one half a head apart and rotate the pair."""
        x1, x2 = x.chunk(2, dim=-1)  # [B, H, T, Dh/2] each
        return torch.cat([-x2, x1], dim=-1)  # [B, H, T, Dh]

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:  # [B, H, T, Dh]
        """Rotate queries or keys by a position-dependent angle."""
        seq_len = x.size(2)
        cos = self.rope_cos[:seq_len]  # [T, Dh]
        sin = self.rope_sin[:seq_len]  # [T, Dh]
        return x * cos + self.rotate_half(x) * sin  # [B, H, T, Dh]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Return attention outputs for one batch of embeddings."""
        seq_len = x.size(1)

        q = self.split_heads(self.q_proj(x), NUM_Q_HEADS)  # [B, Hq, T, Dh]
        q = self.q_norm(q)  # [B, Hq, T, Dh]

        k = self.split_heads(self.k_proj(x), NUM_KV_HEADS)  # [B, Hkv, T, Dh]
        k = self.k_norm(k)  # [B, Hkv, T, Dh]
        k = self.repeat_kv_heads(k)  # [B, Hq, T, Dh]

        if not self.is_global:
            q = self.apply_rope(q)  # [B, Hq, T, Dh]
            k = self.apply_rope(k)  # [B, Hkv, T, Dh]

        v = self.split_heads(self.v_proj(x), NUM_KV_HEADS)  # [B, Hkv, T, Dh]
        v = self.repeat_kv_heads(v)  # [B, Hq, T, Dh]

        attn_scores = (q @ k.mT) / math.sqrt(D_HEAD)  # [B, Hq, T, T]
        causal_mask = self.causal_mask[:seq_len, :seq_len]  # [T, T]
        attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)  # [B, Hq, T, T]

        attn_weights = attn_scores.softmax(dim=-1)  # [B, Hq, T, T]
        attn_output = attn_weights @ v  # [B, Hq, T, Dh]
        attn_output = self.combine_heads(attn_output)  # [B, T, D]

        gate = torch.sigmoid(self.g_proj(x))  # [B, T, D]
        attn_output = gate * attn_output  # [B, T, D]

        return self.o_proj(attn_output)  # [B, T, D]


class FeedForward(nn.Module):
    """Gate one projection of the input by another and project back down."""

    def __init__(self):
        """Create the three linear layers of the gated MLP."""
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_FFN, bias=False)
        self.up_proj = nn.Linear(D_MODEL, D_FFN, bias=False)
        self.down_proj = nn.Linear(D_FFN, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Return the feed-forward block output."""
        gate = F.silu(self.gate_proj(x))  # [B, T, Dff]
        up = self.up_proj(x)  # [B, T, Dff]
        x = gate * up  # [B, T, Dff]
        x = self.down_proj(x)  # [B, T, D]
        return x


class DecoderBlock(nn.Module):
    """Apply one pre-norm attention sublayer and one pre-norm MLP sublayer."""

    def __init__(self):
        """Create the attention, feed-forward, and normalization sublayers."""
        super().__init__()
        self.attn = CausalSelfAttention()
        self.attn_norm = RMSNorm(D_MODEL)
        self.ffn = FeedForward()
        self.ffn_norm = RMSNorm(D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Return the residual output of one decoder block."""
        x = x + self.attn(self.attn_norm(x))  # [B, T, D]
        x = x + self.ffn(self.ffn_norm(x))  # [B, T, D]
        return x


class Decoder(nn.Module):
    """Stack the decoder blocks and normalize the final residual stream."""

    def __init__(self):
        """Create the block stack and the final norm."""
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(NUM_BLOCKS)])
        self.out_norm = RMSNorm(D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Run the full decoder stack."""
        for block in self.blocks:
            x = block(x)  # [B, T, D]
        return self.out_norm(x)  # [B, T, D]


class LanguageModel(nn.Module):
    """Embed tokens, run the decoder, and predict next-token logits."""

    def __init__(self, vocab_size: int):
        """Create the embedding table and the decoder stack."""
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, D_MODEL)
        self.decoder = Decoder()
        nn.init.normal_(self.embed_tokens.weight, std=INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T]
        """Return next-token logits for one batch of token ids."""
        hidden_states = self.embed_tokens(x)  # [B, T, D]
        hidden_states = self.decoder(hidden_states)  # [B, T, D]
        logits = hidden_states @ self.embed_tokens.weight.T  # [B, T, V]
        return logits


def load_text(split: str) -> str:
    """Load one text split from Hugging Face and join it into one string."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    return "\n".join(text for text in dataset[TEXT_COLUMN] if text)


def build_vocab(train_text: str, val_text: str) -> tuple[list[str], dict[str, int]]:
    """Build one character vocabulary from the train and validation text."""
    chars = sorted(set(train_text + val_text))
    stoi = {char: i for i, char in enumerate(chars)}
    return chars, stoi


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """Turn one text string into a tensor of token ids."""
    return torch.tensor([stoi[char] for char in text], dtype=torch.long, device=DEVICE)


def sample_batch(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random input windows and their next-token targets."""
    max_start = tokens.size(0) - CONTEXT_LEN
    starts = torch.randint(max_start, (BATCH_SIZE,), device=DEVICE)  # [B]
    offsets = torch.arange(CONTEXT_LEN, device=DEVICE)  # [T]
    positions = starts[:, None] + offsets[None, :]  # [B, T]
    return tokens[positions], tokens[positions + 1]  # [B, T], [B, T]


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # [B, T, V], [B, T]
    """Compute next-token cross-entropy for one batch."""
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def estimate_loss(model: LanguageModel, tokens: torch.Tensor) -> float:
    """Estimate the loss of one split over a few random batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(EVAL_BATCHES):
        inputs, targets = sample_batch(tokens)
        total_loss += loss_fn(model(inputs), targets).item()
    model.train()
    return total_loss / EVAL_BATCHES


def main() -> None:
    """Train the model and report the loss."""
    torch.manual_seed(SEED)

    train_text = load_text(TRAIN_SPLIT)
    val_text = load_text(VAL_SPLIT)
    chars, stoi = build_vocab(train_text, val_text)
    train_tokens = encode(train_text, stoi)
    val_tokens = encode(val_text, stoi)

    model = LanguageModel(len(chars)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"vocab_size={len(chars)} parameters={sum(p.numel() for p in model.parameters())}")

    start_seconds = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        inputs, targets = sample_batch(train_tokens)
        loss = loss_fn(model(inputs), targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0:
            train_loss = estimate_loss(model, train_tokens)
            val_loss = estimate_loss(model, val_tokens)
            seconds = time.perf_counter() - start_seconds
            print(
                f"step={step} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} seconds={seconds:.1f}"
            )


if __name__ == "__main__":
    main()
