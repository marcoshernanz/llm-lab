import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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


def list_token_shards(root_dir: Path, split: str) -> list[Path]:
    shard_paths = sorted(root_dir.glob(f"{split}_*.npy"))
    if not shard_paths:
        raise FileNotFoundError(f"No {split!r} token shards found under {root_dir}.")
    return shard_paths


def load_token_shard(path: Path) -> jax.Array:
    if not path.exists():
        raise FileNotFoundError(f"Token shard not found at {path}.")

    token_ids = np.load(path)
    if token_ids.ndim != 1:
        raise ValueError(f"Expected a 1D token shard at {path}, got shape {token_ids.shape}.")
    if token_ids.dtype != np.int32:
        token_ids = token_ids.astype(np.int32)
    return jnp.asarray(token_ids)


def load_token_shard_metadata(root_dir: Path) -> dict[str, object]:
    metadata_path = root_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Token shard metadata not found at {metadata_path}.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_token_split_from_shards(
    root_dir: Path,
    split: str,
    *,
    max_shards: int | None = None,
) -> jax.Array:
    if max_shards is not None and max_shards <= 0:
        raise ValueError("max_shards must be positive when provided")

    shard_paths = list_token_shards(root_dir, split)
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]

    shard_arrays = [np.asarray(load_token_shard(path)) for path in shard_paths]
    if not shard_arrays:
        raise ValueError(f"No {split!r} token shards selected from {root_dir}.")

    return jnp.asarray(np.concatenate(shard_arrays, axis=0))
