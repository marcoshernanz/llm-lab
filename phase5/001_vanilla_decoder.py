"""Phase 5 experiment 001: the vanilla decoder-only transformer baseline."""

from __future__ import annotations

import torch
from torch import nn
from datasets import load_dataset  # pyright: ignore

DATASET_NAME = "roneneldan/TinyStories"
DATASET_CONFIG = None
TRAIN_SPLIT = "train[:20000]"
VAL_SPLIT = "validation[:2000]"
TEXT_COLUMN = "text"
DEVICE = "mps"

VOCAB_SIZE = 128
EMBEDDING_DIM = 16
CONTEXT_LEN = 16


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.position_embeddings = nn.Embedding(CONTEXT_LEN, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor):  # [B, T]
        positions = torch.arange(x.size(1), device=DEVICE)  # [T]
        token_embeddings = self.token_embeddings(x)  # [B, T, D]
        position_embeddings = self.position_embeddings(positions)  # [T, D]
        x = token_embeddings + position_embeddings  # [B, T, D]


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
