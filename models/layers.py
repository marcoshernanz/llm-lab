"""Minimal neural network layers used by the learning-focused transformer models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

import math


class LayerNorm(nnx.Module):
    """Normalize features per token with learned scale and bias."""

    weight: nnx.Param[jax.Array]
    bias: nnx.Param[jax.Array]
    eps: float

    def __init__(self, features: int, *, eps: float = 1e-5):
        """Initialize a simple layer normalization module."""
        self.weight = nnx.Param(jnp.ones((features,)))
        self.bias = nnx.Param(jnp.zeros((features,)))
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply layer normalization over the last dimension."""
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + self.eps)
        return self.weight * normalized + self.bias


class Embedding(nnx.Module):
    """Map token or position ids to learned embedding vectors."""

    weight: nnx.Param[jax.Array]

    def __init__(self, num_embeddings: int, embedding_dim: int, *, rngs: nnx.Rngs):
        """Initialize an embedding table with scaled random weights."""
        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight = nnx.Param(rngs.params.normal((num_embeddings, embedding_dim)) * scale)

    def __call__(self, indices: jax.Array) -> jax.Array:
        """Look up embeddings for a batch of indices."""
        return self.weight[indices]


class Linear(nnx.Module):
    """Apply a learned affine projection to the last dimension."""

    weight: nnx.Param[jax.Array]
    bias: nnx.Param[jax.Array] | None

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs, bias: bool = True):
        """Initialize a dense layer with optional bias."""
        scale = 1.0 / math.sqrt(in_features)
        self.weight = nnx.Param(rngs.params.normal((in_features, out_features)) * scale)
        self.bias = nnx.Param(jnp.zeros((out_features,))) if bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        """Project inputs through the learned linear map."""
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output
