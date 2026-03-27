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
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    num_heads: int
    head_dim: int

    def __init__(self, embedding_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        assert embedding_dim % num_heads == 0

        self.q_proj = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.k_proj = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.v_proj = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.out_proj = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

    def split_heads(self, x: jax.Array) -> jax.Array:
        batch_size, sequence_length, _ = x.shape
        return x.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).swapaxes(1, 2)

    def combine_heads(self, x: jax.Array) -> jax.Array:
        batch_size, _, sequence_length, _ = x.shape
        return x.swapaxes(1, 2).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
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
    in_proj: Linear
    out_proj: Linear

    def __init__(self, embedding_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.in_proj = Linear(embedding_dim, hidden_dim, rngs=rngs)
        self.out_proj = Linear(hidden_dim, embedding_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = jax.nn.tanh(self.in_proj(x))
        return self.out_proj(hidden)


class DecoderBlock(nnx.Module):
    self_attention: CausalSelfAttention
    attention_norm: LayerNorm
    feed_forward: FeedForward
    feed_forward_norm: LayerNorm

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.self_attention = CausalSelfAttention(embedding_dim, num_heads, rngs=rngs)
        self.attention_norm = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim, rngs=rngs)
        self.feed_forward_norm = LayerNorm(embedding_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.self_attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Decoder(nnx.Module):
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
        self.blocks = nnx.List(
            [
                DecoderBlock(embedding_dim, hidden_dim, num_heads, rngs=rngs)
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = LayerNorm(embedding_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


class LanguageModel(nnx.Module):
    token_embedding: Embedding
    position_embedding: Embedding
    decoder: Decoder
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_decoder_blocks: int,
        context_length: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.token_embedding = Embedding(vocab_size, embedding_dim, rngs=rngs)
        self.position_embedding = Embedding(context_length, embedding_dim, rngs=rngs)
        self.decoder = Decoder(embedding_dim, hidden_dim, num_heads, num_decoder_blocks, rngs=rngs)
        self.lm_head = Linear(embedding_dim, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        decoder_input = token_embeddings + position_embeddings
        decoder_output = self.decoder(decoder_input)
        return self.lm_head(decoder_output)
