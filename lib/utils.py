import jax
import jax.numpy as jnp

from pathlib import Path
from typing import Callable

from tokenizer.bpe import BPEModel
from models.transformer import DecoderOnlyTransformer


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


def evaluate_split(
    tokens: jax.Array,
    model: DecoderOnlyTransformer,
    loss_fn: Callable[[DecoderOnlyTransformer, jax.Array, jax.Array], jax.Array],
    context_length: int,
    batch_size: int,
) -> float:
    max_start = tokens.shape[0] - context_length
    if max_start <= 0:
        raise ValueError("token sequence must be longer than context_length")

    total_loss = 0.0
    total_examples = 0

    for batch_start in range(0, max_start, batch_size):
        batch_end = min(batch_start + batch_size, max_start)
        start_positions = jnp.arange(batch_start, batch_end, dtype=jnp.int32)
        input_ids, target_ids = build_examples(tokens, start_positions, context_length)
        batch_loss = loss_fn(model, input_ids, target_ids)
        current_batch_size = int(start_positions.shape[0])
        total_loss += float(batch_loss) * current_batch_size
        total_examples += current_batch_size

    return total_loss / total_examples
