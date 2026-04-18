"""Phase 4 experiment 004: a tiny fixed-configuration PyTorch character decoder LM with rotary positional embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from datasets import load_dataset  # pyright: ignore
from torch import nn

DATASET_NAME = "roneneldan/TinyStories"
DATASET_CONFIG = None
TRAIN_SPLIT = "train[:20000]"
VALIDATION_SPLIT = "validation[:2000]"
TEXT_COLUMN = "text"
DEVICE = "mps"
SEED = 1337

SEQUENCE_LEN = 128
EMBEDDING_DIM = 64
NUM_HEADS = 4
assert EMBEDDING_DIM % NUM_HEADS == 0
assert (EMBEDDING_DIM // NUM_HEADS) % 2 == 0
HIDDEN_DIM = 256
NUM_BLOCKS = 4

BATCH_SIZE = 64
LEARNING_RATE = 3e-3
TRAIN_STEPS = 2_000
EVAL_INTERVAL = 200
EVAL_BATCHES = 32


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Rotate query or key vectors by position-dependent angles."""
    _, _, sequence_len, head_dim = x.shape

    positions = torch.arange(sequence_len, dtype=torch.float32, device=x.device)
    pair_ids = torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device)
    angles = positions[:, None] / (10000.0 ** (pair_ids / head_dim))[None, :]
    cos = torch.cos(angles).to(x.dtype)[None, None, :, :]
    sin = torch.sin(angles).to(x.dtype)[None, None, :, :]

    rotated = torch.empty_like(x)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


class CausalSelfAttention(nn.Module):
    """Apply masked multi-head self-attention over one sequence."""

    def __init__(self):
        """Create the query, key, value, and output projections."""
        super().__init__()
        self.query = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.key = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.out = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.num_heads = NUM_HEADS
        self.head_dim = EMBEDDING_DIM // NUM_HEADS

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape embeddings into separate attention heads."""
        batch_size, sequence_len, _ = x.shape
        return x.reshape(batch_size, sequence_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads back into one embedding axis."""
        batch_size, _, sequence_len, _ = x.shape
        return x.swapaxes(1, 2).reshape(batch_size, sequence_len, self.num_heads * self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention outputs for one batch of embeddings."""
        sequence_length = x.shape[1]

        queries = self.split_heads(self.query(x))
        keys = self.split_heads(self.key(x))
        values = self.split_heads(self.value(x))
        queries = apply_rope(queries)
        keys = apply_rope(keys)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(causal_mask, -torch.inf)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = self.combine_heads(attention_weights @ values)
        return self.out(attended_values)


class FeedForward(nn.Module):
    """Project up, apply a nonlinearity, and project back down."""

    def __init__(self):
        """Create the two linear layers of the MLP."""
        super().__init__()
        self.hidden = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feed-forward block output."""
        x = F.gelu(self.hidden(x))
        return self.out(x)


class RMSNorm(nn.Module):
    """Scale activations by their root-mean-square magnitude."""

    def __init__(self):
        """Create the learned scale parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(EMBEDDING_DIM))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one embedding vector and apply the learned scale."""
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class DecoderBlock(nn.Module):
    """Apply one pre-norm attention block followed by one MLP block."""

    def __init__(self):
        """Create the norms, attention, and feed-forward sublayers."""
        super().__init__()
        self.attention_norm = RMSNorm()
        self.attention = CausalSelfAttention()
        self.feed_forward_norm = RMSNorm()
        self.feed_forward = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual output of one decoder block."""
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Decoder(nn.Module):
    """Stack several decoder blocks and finish with one output norm."""

    def __init__(self) -> None:
        """Create the block stack."""
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(NUM_BLOCKS)])
        self.out_norm = RMSNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full decoder stack."""
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class LanguageModel(nn.Module):
    """Embed tokens, apply the decoder, and predict next-token logits."""

    def __init__(self, vocab_size: int):
        """Create the embeddings and decoder."""
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for one batch of token ids."""
        x = self.token_embedding(x)
        x = self.decoder(x)
        x = x @ self.token_embedding.weight.T
        return x


def load_text(split: str) -> str:
    """Load one text split from Hugging Face and join it into one string."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    parts = [text for text in dataset[TEXT_COLUMN] if text]
    text = "\n".join(parts)
    return text


def build_vocab(train_text: str, validation_text: str) -> tuple[list[str], dict[str, int]]:
    """Build one character vocabulary from the train and validation text."""
    vocab_chars = sorted(set(train_text + validation_text))
    char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
    return vocab_chars, char_to_id


def encode_text(text: str, char_to_id: dict[str, int]) -> torch.Tensor:
    """Turn one text string into a tensor of character ids."""
    return torch.tensor([char_to_id[char] for char in text], dtype=torch.long, device=DEVICE)


def sample_batch(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample contiguous input and target windows from one token stream."""
    max_start = token_ids.size(0) - SEQUENCE_LEN
    starts = torch.randint(0, max_start, (BATCH_SIZE,), device=DEVICE)
    offsets = torch.arange(SEQUENCE_LEN, device=DEVICE)
    positions = starts[:, None] + offsets[None, :]
    inputs = token_ids[positions]
    targets = token_ids[positions + 1]
    return inputs, targets


@torch.no_grad()
def estimate_loss(model: LanguageModel, token_ids: torch.Tensor) -> float:
    """Estimate one split loss with a few random batches."""
    losses: list[float] = []
    model.eval()

    for _ in range(EVAL_BATCHES):
        inputs, targets = sample_batch(token_ids)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        losses.append(float(loss.item()))

    model.train()
    return sum(losses) / len(losses)


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross-entropy for one batch."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def main() -> None:
    """Load the dataset, train the tiny model, and print a short report."""
    torch.manual_seed(SEED)

    train_text = load_text(TRAIN_SPLIT)
    validation_text = load_text(VALIDATION_SPLIT)
    vocab_chars, char_to_id = build_vocab(train_text, validation_text)
    train_token_ids = encode_text(train_text, char_to_id)
    validation_token_ids = encode_text(validation_text, char_to_id)

    model = LanguageModel(len(vocab_chars)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(TRAIN_STEPS):
        inputs, targets = sample_batch(train_token_ids)
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_log = step == 0 or (step + 1) % EVAL_INTERVAL == 0 or step == TRAIN_STEPS - 1
        if should_log:
            train_loss = estimate_loss(model, train_token_ids)
            validation_loss = estimate_loss(model, validation_token_ids)
            print(
                f"step={step + 1} "
                f"batch_loss={loss.item():.4f} "
                f"train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f}"
            )


if __name__ == "__main__":
    main()
