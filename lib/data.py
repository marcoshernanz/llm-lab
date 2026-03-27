from pathlib import Path

import jax
import jax.numpy as jnp

from tokenizer.bpe import BPEModel


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")

    text = path.read_text(encoding="utf-8")
    if len(text) == 0:
        raise ValueError("Dataset is empty.")

    return text


def load_tokenizer(path: Path) -> BPEModel:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found at {path}.")

    return BPEModel.load(path)


def build_token_splits(
    text: str,
    tokenizer: BPEModel,
    train_split: float,
) -> tuple[jax.Array, jax.Array]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")

    split_index = int(len(text) * train_split)
    train_text = text[:split_index]
    validation_text = text[split_index:]
    train_token_ids = jnp.asarray(tokenizer.encode(train_text), dtype=jnp.int32)
    validation_token_ids = jnp.asarray(tokenizer.encode(validation_text), dtype=jnp.int32)
    return train_token_ids, validation_token_ids


def build_examples(
    token_ids: jax.Array,
    start_positions: jax.Array,
    context_length: int,
) -> tuple[jax.Array, jax.Array]:
    offsets = jnp.arange(context_length, dtype=start_positions.dtype)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions[:, None] + offsets + 1]
    return input_ids, target_ids
