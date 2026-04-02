"""Implement minimal learning-oriented optimizers."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp


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


def init_velocity(model: nnx.Module) -> nnx.State[Any, Any]:
    """Create a zero velocity tree that matches the model parameter tree."""
    params = nnx.state(model, nnx.Param)
    return jax.tree.map(jnp.zeros_like, params)


def apply_sgd_momentum(
    model: nnx.Module,
    grads: nnx.State[Any, Any],
    velocity: nnx.State[Any, Any],
    learning_rate: float,
    momentum: float,
) -> nnx.State[Any, Any]:
    """Apply one momentum-SGD update and return the new velocity tree."""
    params = nnx.state(model, nnx.Param)

    def update_velocity(velocity_leaf: jax.Array, grad: jax.Array) -> jax.Array:
        """Update one velocity leaf using the current gradient leaf."""
        return momentum * velocity_leaf - learning_rate * grad

    new_velocity = jax.tree.map(update_velocity, velocity, grads)
    new_params = jax.tree.map(lambda param, velocity_leaf: param + velocity_leaf, params, new_velocity)
    nnx.update(model, new_params)
    return new_velocity
