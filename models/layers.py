from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

import math


class LayerNorm(nnx.Module):
    weight: nnx.Param[jax.Array]
    bias: nnx.Param[jax.Array]
    eps: float

    def __init__(self, features: int, *, eps: float = 1e-5):
        self.weight = nnx.Param(jnp.ones((features,)))
        self.bias = nnx.Param(jnp.zeros((features,)))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + self.eps)
        return self.weight * normalized + self.bias


class Embedding(nnx.Module):
    weight: nnx.Param[jax.Array]

    def __init__(self, num_embeddings: int, embedding_dim: int, *, rngs: nnx.Rngs):
        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight = nnx.Param(rngs.params.normal((num_embeddings, embedding_dim)) * scale)

    def __call__(self, indices: jax.Array) -> jax.Array:
        return self.weight[indices]


class Linear(nnx.Module):
    weight: nnx.Param[jax.Array]
    bias: nnx.Param[jax.Array] | None

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs, bias: bool = True):
        scale = 1.0 / math.sqrt(in_features)
        self.weight = nnx.Param(rngs.params.normal((in_features, out_features)) * scale)
        self.bias = nnx.Param(jnp.zeros((out_features,))) if bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output
