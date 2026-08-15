"""Phase 5 experiment 001: the vanilla decoder-only transformer baseline."""

from __future__ import annotations

import torch
from datasets import load_dataset  # pyright: ignore

DATASET_NAME = "roneneldan/TinyStories"
DATASET_CONFIG = None
TRAIN_SPLIT = "train[:20000]"
VALIDATION_SPLIT = "validation[:2000]"
TEXT_COLUMN = "text"
DEVICE = "mps"


def load_text(split: str) -> str:
    """Load one text split from Hugging Face and join it into one string."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    return "\n".join(text for text in dataset[TEXT_COLUMN] if text)


def build_vocab(train_text: str, validation_text: str) -> tuple[list[str], dict[str, int]]:
    """Build one character vocabulary from the train and validation text."""
    vocab_chars = sorted(set(train_text + validation_text))
    char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
    return vocab_chars, char_to_id


def encode_text(text: str, char_to_id: dict[str, int]) -> torch.Tensor:
    """Turn one text string into a tensor of character ids."""
    return torch.tensor([char_to_id[char] for char in text], dtype=torch.long, device=DEVICE)


def main() -> None:
    """Load the dataset and report the character vocabulary and token counts."""
    train_text = load_text(TRAIN_SPLIT)
    validation_text = load_text(VALIDATION_SPLIT)
    vocab_chars, char_to_id = build_vocab(train_text, validation_text)
    train_token_ids = encode_text(train_text, char_to_id)
    validation_token_ids = encode_text(validation_text, char_to_id)

    print(f"vocab_size={len(vocab_chars)}")
    print(f"train_tokens={train_token_ids.numel()}")
    print(f"validation_tokens={validation_token_ids.numel()}")


if __name__ == "__main__":
    main()
