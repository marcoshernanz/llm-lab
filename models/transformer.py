"""Small decoder-only transformer pieces built for first-principles study."""

from __future__ import annotations

import math

from flax import nnx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from models.layers import Embedding
from models.layers import LayerNorm
from models.layers import Linear


class CausalSelfAttention(nnx.Module):
    """Compute masked self-attention over a token sequence."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    num_heads: int
    head_dim: int

    def __init__(self, embedding_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        """Initialize query, key, value, and output projections."""
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.q_proj = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.k_proj = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.v_proj = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.out_proj = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

    def split_heads(self, x: jax.Array) -> jax.Array:
        """Reshape embeddings into separate attention heads."""
        batch_size, sequence_length, _ = x.shape
        return x.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).swapaxes(1, 2)

    def combine_heads(self, x: jax.Array) -> jax.Array:
        """Merge per-head outputs back into the embedding dimension."""
        batch_size, _, sequence_length, _ = x.shape
        return x.swapaxes(1, 2).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal multi-head self-attention to the input sequence."""
        sequence_length = x.shape[1]

        queries = self.split_heads(self.q_proj(x))
        keys = self.split_heads(self.k_proj(x))
        values = self.split_heads(self.v_proj(x))

        attention_scores = (queries @ keys.mT) / math.sqrt(self.head_dim)
        causal_mask = jnp.triu(jnp.ones((sequence_length, sequence_length), dtype=bool), k=1)
        attention_scores = jnp.where(causal_mask, -jnp.inf, attention_scores)

        attention_weights = jnn.softmax(attention_scores, axis=-1)
        attended_values = attention_weights @ values
        return self.out_proj(self.combine_heads(attended_values))


class FeedForward(nnx.Module):
    """Apply the position-wise MLP inside a decoder block."""

    in_proj: Linear
    out_proj: Linear

    def __init__(self, embedding_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        """Initialize the two linear layers of the feed-forward block."""
        self.in_proj = Linear(embedding_dim, hidden_dim, rngs=rngs)
        self.out_proj = Linear(hidden_dim, embedding_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Project up, apply a nonlinearity, and project back down."""
        hidden = jnn.gelu(self.in_proj(x))
        return self.out_proj(hidden)


class DecoderBlock(nnx.Module):
    """Combine pre-norm attention and feed-forward residual sublayers."""

    self_attention: CausalSelfAttention
    attention_norm: LayerNorm
    feed_forward: FeedForward
    feed_forward_norm: LayerNorm

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        """Initialize one decoder block."""
        self.self_attention = CausalSelfAttention(embedding_dim, num_heads, rngs=rngs)
        self.attention_norm = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim, rngs=rngs)
        self.feed_forward_norm = LayerNorm(embedding_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run one residual decoder block."""
        x = x + self.self_attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Decoder(nnx.Module):
    """Stack multiple decoder blocks and finish with output normalization."""

    blocks: nnx.List[DecoderBlock]
    output_norm: LayerNorm

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize a decoder stack with the requested depth."""
        self.blocks = nnx.List(
            [
                DecoderBlock(embedding_dim, hidden_dim, num_heads, rngs=rngs)
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = LayerNorm(embedding_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply every decoder block in sequence."""
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


class DecoderOnlyTransformer(nnx.Module):
    """Predict next-token logits from token and position embeddings."""

    token_embedding: Embedding
    position_embedding: Embedding
    decoder_stack: Decoder

    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_decoder_blocks: int,
        rngs: nnx.Rngs,
    ):
        """Initialize token embeddings, position embeddings, and decoder blocks."""

        self.token_embedding = Embedding(vocab_size, embedding_dim, rngs=rngs)
        self.position_embedding = Embedding(context_length, embedding_dim, rngs=rngs)
        self.decoder_stack = Decoder(
            embedding_dim, hidden_dim, num_heads, num_decoder_blocks, rngs=rngs
        )

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """Return next-token logits for a batch of token ids."""
        if input_ids.shape[-1] > self.position_embedding.weight.shape[0]:
            raise ValueError("input sequence length exceeds context length")

        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.decoder_stack(x)
        return x @ self.token_embedding.weight.T
