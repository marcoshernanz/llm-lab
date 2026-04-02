"""Implement minimal learning-oriented optimizers."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp


def apply_sgd(
    model: nnx.Module,
    grads: nnx.State[Any, Any],
    learning_rate: float,
) -> None:
    """Apply one plain SGD update to the model parameters."""
    params = nnx.state(model, nnx.Param)
    new_params = jax.tree.map(lambda param, grad: param - learning_rate * grad, params, grads)
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
    new_velocity = jax.tree.map(
        lambda velocity_leaf, grad: momentum * velocity_leaf - learning_rate * grad,
        velocity,
        grads,
    )
    new_params = jax.tree.map(
        lambda param, velocity_leaf: param + velocity_leaf, params, new_velocity
    )
    nnx.update(model, new_params)
    return new_velocity


def init_adam_state(
    model: nnx.Module,
) -> tuple[nnx.State[Any, Any], nnx.State[Any, Any], jax.Array]:
    """Create zero Adam moment trees and the initial step counter."""
    params = nnx.state(model, nnx.Param)
    first_moment = jax.tree.map(jnp.zeros_like, params)
    second_moment = jax.tree.map(jnp.zeros_like, params)
    step = jnp.array(0, dtype=jnp.int32)
    return first_moment, second_moment, step


def apply_adam(
    model: nnx.Module,
    grads: nnx.State[Any, Any],
    first_moment: nnx.State[Any, Any],
    second_moment: nnx.State[Any, Any],
    step: jax.Array,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[nnx.State[Any, Any], nnx.State[Any, Any], jax.Array]:
    """Apply one Adam step and return the updated optimizer state."""
    params = nnx.state(model, nnx.Param)
    step = step + 1

    new_first_moment = jax.tree.map(
        lambda first_leaf, grad: beta1 * first_leaf + (1.0 - beta1) * grad,
        first_moment,
        grads,
    )
    new_second_moment = jax.tree.map(
        lambda second_leaf, grad: beta2 * second_leaf + (1.0 - beta2) * jnp.square(grad),
        second_moment,
        grads,
    )

    corrected_first_moment = jax.tree.map(
        lambda first_leaf: first_leaf / (1.0 - jnp.power(beta1, step)),
        new_first_moment,
    )
    corrected_second_moment = jax.tree.map(
        lambda second_leaf: second_leaf / (1.0 - jnp.power(beta2, step)),
        new_second_moment,
    )
    new_params = jax.tree.map(
        lambda param, first_leaf, second_leaf: (
            param - learning_rate * first_leaf / (jnp.sqrt(second_leaf) + epsilon)
        ),
        params,
        corrected_first_moment,
        corrected_second_moment,
    )
    nnx.update(model, new_params)
    return new_first_moment, new_second_moment, step


def apply_adamw(
    model: nnx.Module,
    grads: nnx.State[Any, Any],
    first_moment: nnx.State[Any, Any],
    second_moment: nnx.State[Any, Any],
    step: jax.Array,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
) -> tuple[nnx.State[Any, Any], nnx.State[Any, Any], jax.Array]:
    """Apply one AdamW step and return the updated optimizer state."""
    params = nnx.state(model, nnx.Param)
    step = step + 1

    new_first_moment = jax.tree.map(
        lambda first_leaf, grad: beta1 * first_leaf + (1.0 - beta1) * grad,
        first_moment,
        grads,
    )
    new_second_moment = jax.tree.map(
        lambda second_leaf, grad: beta2 * second_leaf + (1.0 - beta2) * jnp.square(grad),
        second_moment,
        grads,
    )

    corrected_first_moment = jax.tree.map(
        lambda first_leaf: first_leaf / (1.0 - jnp.power(beta1, step)),
        new_first_moment,
    )
    corrected_second_moment = jax.tree.map(
        lambda second_leaf: second_leaf / (1.0 - jnp.power(beta2, step)),
        new_second_moment,
    )
    new_params = jax.tree.map(
        lambda param, first_leaf, second_leaf: (
            param - learning_rate * first_leaf / (jnp.sqrt(second_leaf) + epsilon)
        ),
        params,
        corrected_first_moment,
        corrected_second_moment,
    )
    nnx.update(model, new_params)
    return new_first_moment, new_second_moment, step
