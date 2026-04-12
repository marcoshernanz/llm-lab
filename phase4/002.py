"""Phase 4 baseline: a tiny fixed-configuration PyTorch character LM trainer."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset  # pyright: ignore

DATASET_NAME = "roneneldan/TinyStories"
DATASET_CONFIG = None
TRAIN_SPLIT = "train[:20000]"
VALIDATION_SPLIT = "validation[:2000]"
TEXT_COLUMN = "text"
DEVICE = "mps"
SEED = 1337

VOCAB_SIZE = 128
SEQUENCE_LEN = 128
EMBEDDING_DIM = 64
NUM_HEADS = 4
assert EMBEDDING_DIM % NUM_HEADS == 0

BATCH_SIZE = 64
LEARNING_RATE = 3e-3
TRAIN_STEPS = 2_000
EVAL_INTERVAL = 200
EVAL_BATCHES = 32
SAMPLE_LENGTH = 400


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.key = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.value = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.num_heads = NUM_HEADS
        self.head_dim = EMBEDDING_DIM // NUM_HEADS

    def split_heads(self, x: torch.Tensor):
        batch_size, sequence_len, _ = x.shape
        return x.reshape(batch_size, sequence_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def combine_heads(self, x: torch.Tensor):
        batch_size, _, sequence_len, _ = x.shape
        return x.swapaxes(1, 2).reshape(batch_size, sequence_len, self.num_heads * self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]

        queries = self.split_heads(self.query(x))
        keys = self.split_heads(self.key(x))
        values = self.split_heads(self.value(x))

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(causal_mask, -torch.inf)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = attention_weights @ values
        return self.combine_heads(attended_values)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_embedding = nn.Embedding(SEQUENCE_LEN, EMBEDDING_DIM)
        # Attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x) + self.position_embedding(x)

        return x


def load_text(split: str) -> str:
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    parts = [text for text in dataset[TEXT_COLUMN] if text]
    text = "\n".join(parts)
    return text


def build_vocab(train_text: str, validation_text: str) -> tuple[list[str], dict[str, int]]:
    vocab_chars = sorted(set(train_text + validation_text))
    char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
    return vocab_chars, char_to_id


def encode_text(text: str, char_to_id: dict[str, int]) -> torch.Tensor:
    return torch.tensor([char_to_id[char] for char in text], dtype=torch.long, device=DEVICE)


def sample_batch(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = token_ids.size(0) - SEQUENCE_LEN
    starts = torch.randint(0, max_start, (BATCH_SIZE,), device=DEVICE)
    offsets = torch.arange(SEQUENCE_LEN, device=DEVICE)
    positions = starts[:, None] + offsets[None, :]
    inputs = token_ids[positions]
    targets = token_ids[positions + 1]
    return inputs, targets


@torch.no_grad()
def estimate_loss(model: LanguageModel, token_ids: torch.Tensor) -> float:
    losses: list[float] = []
    model.eval()

    for _ in range(EVAL_BATCHES):
        inputs, targets = sample_batch(token_ids)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        losses.append(float(loss.item()))

    model.train()
    return sum(losses) / len(losses)


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
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

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
