"""Implement minimal learning-oriented optimizers."""

from typing import Any

from flax import nnx
import jax


def apply_sgd(
    model: nnx.Module,
    grads: nnx.State[Any, Any],
    learning_rate: float,
) -> None:
    """Apply one plain SGD update to the model parameters."""
    params = nnx.state(model, nnx.Param)
    new_params = jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)
    nnx.update(model, new_params)
