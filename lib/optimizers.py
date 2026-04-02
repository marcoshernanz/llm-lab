"""Implement minimal learning-oriented optimizers."""

from typing import Any

from flax import nnx
import jax


def sgd_update(
    param: jax.Array,
    grad: jax.Array,
    learning_rate: float,
) -> jax.Array:
    """Return the plain SGD parameter update for one array leaf."""
    return param - learning_rate * grad


def apply_sgd(
    model: nnx.Module,
    grads: nnx.State[Any, Any],
    learning_rate: float,
) -> None:
    """Apply one plain SGD update to the model parameters."""
    params = nnx.state(model, nnx.Param)

    def update_leaf(param: jax.Array, grad: jax.Array) -> jax.Array:
        """Apply the SGD rule to one parameter leaf and its gradient."""
        return sgd_update(param, grad, learning_rate)

    new_params = jax.tree.map(update_leaf, params, grads)
    nnx.update(model, new_params)
