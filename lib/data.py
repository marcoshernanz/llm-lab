"""Load text, tokenizers, and token shards for small language-model experiments."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from tokenizer.bpe import BPEModel


def load_text(path: Path) -> str:
    """Read a UTF-8 text corpus and validate that it is non-empty."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")

    text = path.read_text(encoding="utf-8")
    if len(text) == 0:
        raise ValueError("Dataset is empty.")

    return text


def load_tokenizer(path: Path) -> BPEModel:
    """Load a saved tokenizer artifact used by the experiments."""
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found at {path}.")

    return BPEModel.load(path)


def build_token_splits(
    text: str,
    tokenizer: BPEModel,
    train_split: float,
) -> tuple[jax.Array, jax.Array]:
    """Tokenize text and split it into train and validation sequences."""
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
    """Build input and target windows for next-token prediction."""
    offsets = jnp.arange(context_length, dtype=start_positions.dtype)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions[:, None] + offsets + 1]
    return input_ids, target_ids


def list_token_shards(root_dir: Path, split: str) -> list[Path]:
    """List token shard files for one dataset split."""
    shard_paths = sorted(root_dir.glob(f"{split}_*.npy"))
    if not shard_paths:
        raise FileNotFoundError(f"No {split!r} token shards found under {root_dir}.")
    return shard_paths


def load_token_shard(path: Path, *, mmap: bool = False) -> jax.Array:
    """Load one token shard with JAX and normalize it to int32."""
    if not path.exists():
        raise FileNotFoundError(f"Token shard not found at {path}.")

    token_ids = jnp.load(path, mmap_mode="r" if mmap else None)
    if token_ids.ndim != 1:
        raise ValueError(f"Expected a 1D token shard at {path}, got shape {token_ids.shape}.")
    return token_ids.astype(jnp.int32)


def load_token_shard_metadata(root_dir: Path) -> dict[str, object]:
    """Read the metadata that describes a token shard dataset."""
    metadata_path = root_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Token shard metadata not found at {metadata_path}.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_token_split_from_shards(
    root_dir: Path,
    split: str,
    *,
    max_shards: int | None = None,
    mmap: bool = False,
) -> jax.Array:
    """Concatenate one split from multiple shard files into one array."""
    if max_shards is not None and max_shards <= 0:
        raise ValueError("max_shards must be positive when provided")

    shard_paths = list_token_shards(root_dir, split)
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]

    shard_arrays = [load_token_shard(path, mmap=mmap) for path in shard_paths]
    if not shard_arrays:
        raise ValueError(f"No {split!r} token shards selected from {root_dir}.")

    return jnp.concatenate(shard_arrays, axis=0)
