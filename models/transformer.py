from __future__ import annotations

import math

from flax import nnx
import jax
import jax.nn as jnn
import jax.numpy as jnp

from models.layers import Embedding
from models.layers import LayerNorm
from models.layers import Linear
from training.config import TrainingConfig


LAYER_NORM_EPS = 1e-5


class CausalSelfAttention(nnx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear
    num_heads: int
    head_dim: int

    def __init__(self, embedding_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.key = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.value = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)
        self.output = Linear(embedding_dim, embedding_dim, rngs=rngs, bias=False)

    def split_heads(self, x: jax.Array) -> jax.Array:
        batch_size, sequence_length, _ = x.shape
        head_shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        return x.reshape(head_shape).swapaxes(1, 2)

    def combine_heads(self, x: jax.Array) -> jax.Array:
        batch_size, _, sequence_length, _ = x.shape
        combined_shape = (batch_size, sequence_length, self.num_heads * self.head_dim)
        return x.swapaxes(1, 2).reshape(combined_shape)

    def __call__(self, x: jax.Array) -> jax.Array:
        sequence_length = x.shape[1]
        queries = self.split_heads(self.query(x))
        keys = self.split_heads(self.key(x))
        values = self.split_heads(self.value(x))

        attention_scores = (queries @ keys.mT) / math.sqrt(self.head_dim)
        causal_mask = jnp.triu(jnp.ones((sequence_length, sequence_length), dtype=bool), k=1)
        masked_attention_scores = jnp.where(causal_mask, -jnp.inf, attention_scores)
        attention_weights = jnn.softmax(masked_attention_scores, axis=-1)
        attended_values = attention_weights @ values
        combined_heads = self.combine_heads(attended_values)
        return self.output(combined_heads)


class FeedForward(nnx.Module):
    hidden: Linear
    output: Linear

    def __init__(self, embedding_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.hidden = Linear(embedding_dim, hidden_dim, rngs=rngs)
        self.output = Linear(hidden_dim, embedding_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden_activation = jnp.tanh(self.hidden(x))
        return self.output(hidden_activation)


class DecoderBlock(nnx.Module):
    attention: CausalSelfAttention
    attention_norm: LayerNorm
    feed_forward: FeedForward
    feed_forward_norm: LayerNorm

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.attention = CausalSelfAttention(embedding_dim, num_heads, rngs=rngs)
        self.attention_norm = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim, rngs=rngs)
        self.feed_forward_norm = LayerNorm(embedding_dim)

    def __call__(self, x: jax.Array, *, eps: float) -> jax.Array:
        attention_residual = x + self.attention(x)
        attention_block_output = self.attention_norm(attention_residual, eps=eps)

        feed_forward_residual = attention_block_output + self.feed_forward(attention_block_output)
        return self.feed_forward_norm(feed_forward_residual, eps=eps)


class Decoder(nnx.Module):
    blocks: nnx.List[DecoderBlock]

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

    def __call__(self, x: jax.Array, *, eps: float) -> jax.Array:
        for block in self.blocks:
            x = block(x, eps=eps)
        return x


class LanguageModel(nnx.Module):
    token_embedding: Embedding
    position_embedding: Embedding
    decoder: Decoder
    lm_head: Linear

    def __init__(self, config: TrainingConfig, vocab_size: int, *, rngs: nnx.Rngs):
        self.token_embedding = Embedding(vocab_size, config.model.embedding_dim, rngs=rngs)
        self.position_embedding = Embedding(
            config.data.context_tokens,
            config.model.embedding_dim,
            rngs=rngs,
        )
        self.decoder = Decoder(
            config.model.embedding_dim,
            config.model.hidden_dim,
            config.model.num_heads,
            config.model.num_decoder_blocks,
            rngs=rngs,
        )
        self.lm_head = Linear(config.model.embedding_dim, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        decoder_input = token_embeddings + position_embeddings
        decoder_output = self.decoder(decoder_input, eps=LAYER_NORM_EPS)
        return self.lm_head(decoder_output)
