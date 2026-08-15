"""Phase 5 experiment 001: the vanilla decoder-only transformer baseline."""

from __future__ import annotations

import math
import torch
from torch import nn
from datasets import load_dataset  # pyright: ignore

DATASET_NAME = "roneneldan/TinyStories"
DATASET_CONFIG = None
TRAIN_SPLIT = "train[:20000]"
VAL_SPLIT = "validation[:2000]"
TEXT_COLUMN = "text"
DEVICE = "mps"

D_MODEL = 16
CONTEXT_LEN = 16


class CausalSelfAttention(nn.Module):
    """Apply masked self-attention over one sequence."""

    def __init__(self):
        """Create the projections and the causal mask."""
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)
        mask = torch.ones(CONTEXT_LEN, CONTEXT_LEN, dtype=torch.bool).triu(diagonal=1)  # [T, T]
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        """Return attention outputs for one batch of embeddings."""
        seq_len = x.size(1)
        q = self.q_proj(x)  # [B, T, D]
        k = self.k_proj(x)  # [B, T, D]
        v = self.v_proj(x)  # [B, T, D]

        attn_scores = (q @ k.mT) / math.sqrt(D_MODEL)  # [B, T, T]
        attn_scores = attn_scores.masked_fill(self.causal_mask[:seq_len, :seq_len], -torch.inf)

        attn_weights = attn_scores.softmax(dim=-1)  # [B, T, T]
        attn_output = attn_weights @ v  # [B, T, D]
        return self.o_proj(attn_output)  # [B, T, D]


class Model(nn.Module):
    """Embed tokens and their positions."""

    def __init__(self, vocab_size: int):
        """Create the token and position embedding tables."""
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, D_MODEL)
        self.embed_positions = nn.Embedding(CONTEXT_LEN, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T]
        """Return token-plus-position embeddings for one batch of token ids."""
        positions = torch.arange(x.size(1), device=x.device)  # [T]
        token_embeddings = self.embed_tokens(x)  # [B, T, D]
        position_embeddings = self.embed_positions(positions)  # [T, D]
        return token_embeddings + position_embeddings  # [B, T, D]


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


def main() -> None:
    """Load the dataset and report the vocabulary size and token counts."""
    train_text = load_text(TRAIN_SPLIT)
    val_text = load_text(VAL_SPLIT)
    chars, stoi = build_vocab(train_text, val_text)
    train_tokens = encode(train_text, stoi)
    val_tokens = encode(val_text, stoi)

    print(f"vocab_size={len(chars)}")
    print(f"train_tokens={train_tokens.numel()}")
    print(f"val_tokens={val_tokens.numel()}")


if __name__ == "__main__":
    main()
