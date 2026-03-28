from typing import Callable

import jax
import jax.numpy as jnp

from lib.data import build_examples
from models.transformer import DecoderOnlyTransformer


def sample_evaluation_positions(
    tokens: jax.Array,
    *,
    context_length: int,
    subset_size: int,
    rng: jax.Array,
) -> jax.Array:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")

    max_start = tokens.shape[0] - context_length
    if max_start <= 0:
        raise ValueError("token sequence must be longer than context_length")

    num_positions = min(subset_size, max_start)
    return jax.random.choice(
        rng,
        max_start,
        shape=(num_positions,),
        replace=False,
    ).astype(jnp.int32)


def evaluate_positions(
    tokens: jax.Array,
    start_positions: jax.Array,
    model: DecoderOnlyTransformer,
    loss_fn: Callable[[DecoderOnlyTransformer, jax.Array, jax.Array], jax.Array],
    context_length: int,
    batch_size: int,
) -> float:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if start_positions.shape[0] == 0:
        raise ValueError("start_positions must contain at least one position")

    total_loss = 0.0
    total_examples = 0

    for batch_start in range(0, start_positions.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, start_positions.shape[0])
        batch_positions = start_positions[batch_start:batch_end]
        input_ids, target_ids = build_examples(tokens, batch_positions, context_length)
        batch_loss = loss_fn(model, input_ids, target_ids)
        current_batch_size = int(batch_positions.shape[0])
        total_loss += float(batch_loss) * current_batch_size
        total_examples += current_batch_size

    return total_loss / total_examples


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

    start_positions = jnp.arange(max_start, dtype=jnp.int32)
    return evaluate_positions(
        tokens,
        start_positions,
        model,
        loss_fn,
        context_length,
        batch_size,
    )
