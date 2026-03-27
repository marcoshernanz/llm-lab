from typing import Callable

import jax
import jax.numpy as jnp

from lib.data import build_examples
from models.transformer import DecoderOnlyTransformer


def evaluate_split(
    tokens: jax.Array,
    model: DecoderOnlyTransformer,
    loss_fn: Callable[[DecoderOnlyTransformer, jax.Array, jax.Array], jax.Array],
    context_length: int,
    batch_size: int,
) -> float:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

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
