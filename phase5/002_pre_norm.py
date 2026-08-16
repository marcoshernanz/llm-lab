"""Phase 5 experiment 001: the vanilla decoder-only transformer baseline."""

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
# H: number of attention heads
# Dh: head dim, D // H
# Dff: feed-forward dim

CONTEXT_LEN = 256
D_MODEL = 128
NUM_HEADS = 4
assert D_MODEL % NUM_HEADS == 0
HEAD_DIM = D_MODEL // NUM_HEADS
D_FFN = 4 * D_MODEL
NUM_BLOCKS = 8
INIT_STD = 0.02

BATCH_SIZE = 32
LEARNING_RATE = 3e-3
GRAD_CLIP_NORM = 1.0
TRAIN_STEPS = 3_000
EVAL_INTERVAL = 250
EVAL_BATCHES = 32


class LayerNorm(nn.Module):
    """Normalize each embedding vector and apply a learned scale and shift."""

    def __init__(self):
        """Create the learned scale and shift parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D_MODEL))
        self.bias = nn.Parameter(torch.zeros(D_MODEL))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Return the normalized, scaled, and shifted embeddings."""
        mean = x.mean(dim=-1, keepdim=True)  # [B, T, 1]
        variance = x.var(dim=-1, keepdim=True, correction=0)  # [B, T, 1]
        normalized = (x - mean) / torch.sqrt(variance + self.eps)  # [B, T, D]
        return normalized * self.weight + self.bias  # [B, T, D]


class CausalSelfAttention(nn.Module):
    """Apply masked self-attention over one sequence."""

    causal_mask: torch.Tensor

    def __init__(self):
        """Create the projections and the causal mask."""
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)
        mask = torch.ones(CONTEXT_LEN, CONTEXT_LEN, dtype=torch.bool).triu(diagonal=1)  # [T, T]
        self.register_buffer("causal_mask", mask)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Split the embedding axis into separate attention heads."""
        batch_size, seq_len, _ = x.size()
        x = x.reshape(batch_size, seq_len, NUM_HEADS, HEAD_DIM)  # [B, T, H, Dh]
        return x.transpose(1, 2)  # [B, H, T, Dh]

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:  # [B, H, T, Dh]
        """Merge the attention heads back into one embedding axis."""
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # [B, T, H, Dh]
        return x.reshape(batch_size, seq_len, D_MODEL)  # [B, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Return attention outputs for one batch of embeddings."""
        seq_len = x.size(1)
        q = self.split_heads(self.q_proj(x))  # [B, H, T, Dh]
        k = self.split_heads(self.k_proj(x))  # [B, H, T, Dh]
        v = self.split_heads(self.v_proj(x))  # [B, H, T, Dh]

        attn_scores = (q @ k.mT) / math.sqrt(HEAD_DIM)  # [B, H, T, T]
        attn_scores = attn_scores.masked_fill(self.causal_mask[:seq_len, :seq_len], -torch.inf)

        attn_weights = attn_scores.softmax(dim=-1)  # [B, H, T, T]
        attn_output = attn_weights @ v  # [B, H, T, Dh]
        attn_output = self.combine_heads(attn_output)  # [B, T, D]
        return self.o_proj(attn_output)  # [B, T, D]


class FeedForward(nn.Module):
    """Project up, apply a nonlinearity, and project back down."""

    def __init__(self):
        """Create the two linear layers of the MLP."""
        super().__init__()
        self.up_proj = nn.Linear(D_MODEL, D_FFN)
        self.down_proj = nn.Linear(D_FFN, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Return the feed-forward block output."""
        x = self.up_proj(x)  # [B, T, Dff]
        x = F.gelu(x)  # [B, T, Dff]
        x = self.down_proj(x)  # [B, T, D]
        return x


class DecoderBlock(nn.Module):
    """Apply one post-norm attention sublayer and one post-norm MLP sublayer."""

    def __init__(self):
        """Create the attention, feed-forward, and normalization sublayers."""
        super().__init__()
        self.attn = CausalSelfAttention()
        self.attn_norm = LayerNorm()
        self.ffn = FeedForward()
        self.ffn_norm = LayerNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Return the residual output of one decoder block."""
        x = x + self.attn(self.attn_norm(x))  # [B, T, D]
        x = x + self.ffn(self.ffn_norm(x))  # [B, T, D]
        return x


class Decoder(nn.Module):
    """Stack the decoder blocks."""

    def __init__(self):
        """Create the block stack."""
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(NUM_BLOCKS)])
        self.out_norm = LayerNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Run the full decoder stack."""
        for block in self.blocks:
            x = block(x)  # [B, T, D]
        return self.out_norm(x)  # [B, T, D]


class LanguageModel(nn.Module):
    """Embed tokens, run the decoder, and predict next-token logits."""

    def __init__(self, vocab_size: int):
        """Create the embedding tables and the decoder stack."""
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, D_MODEL)
        self.embed_positions = nn.Embedding(CONTEXT_LEN, D_MODEL)
        self.decoder = Decoder()
        nn.init.normal_(self.embed_tokens.weight, std=INIT_STD)
        nn.init.normal_(self.embed_positions.weight, std=INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T]
        """Return next-token logits for one batch of token ids."""
        positions = torch.arange(x.size(1), device=x.device)  # [T]
        token_embeddings = self.embed_tokens(x)  # [B, T, D]
        position_embeddings = self.embed_positions(positions)  # [T, D]
        hidden_states = token_embeddings + position_embeddings  # [B, T, D]
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
    """Train the vanilla decoder and report the loss."""
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
