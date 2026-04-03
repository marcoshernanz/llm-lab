"""Decoder-only transformer built from standard NNX primitives."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp


class CausalSelfAttention(nnx.Module):
    """Compute causal self-attention with the built-in NNX attention block."""

    attention: nnx.MultiHeadAttention

    def __init__(self, embedding_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        """Initialize one causal multi-head attention module."""
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embedding_dim,
            qkv_features=embedding_dim,
            out_features=embedding_dim,
            use_bias=False,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal multi-head self-attention to the input sequence."""
        causal_mask = nnx.make_causal_mask(jnp.ones(x.shape[:2], dtype=jnp.bool_))
        return self.attention(x, mask=causal_mask)


class FeedForward(nnx.Module):
    """Apply the position-wise MLP inside a decoder block."""

    in_proj: nnx.Linear
    out_proj: nnx.Linear

    def __init__(self, embedding_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        """Initialize the two linear layers of the feed-forward block."""
        self.in_proj = nnx.Linear(
            in_features=embedding_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            in_features=hidden_dim,
            out_features=embedding_dim,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Project up, apply a nonlinearity, and project back down."""
        hidden = nnx.gelu(self.in_proj(x))
        return self.out_proj(hidden)


class DecoderBlock(nnx.Module):
    """Combine pre-norm attention and feed-forward residual sublayers."""

    self_attention: CausalSelfAttention
    attention_norm: nnx.LayerNorm
    feed_forward: FeedForward
    feed_forward_norm: nnx.LayerNorm

    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        """Initialize one decoder block."""
        self.self_attention = CausalSelfAttention(embedding_dim, num_heads, rngs=rngs)
        self.attention_norm = nnx.LayerNorm(num_features=embedding_dim, rngs=rngs)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim, rngs=rngs)
        self.feed_forward_norm = nnx.LayerNorm(num_features=embedding_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run one residual decoder block."""
        x = x + self.self_attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Decoder(nnx.Module):
    """Stack multiple decoder blocks and finish with output normalization."""

    blocks: nnx.List[DecoderBlock]
    output_norm: nnx.LayerNorm

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
        self.output_norm = nnx.LayerNorm(num_features=embedding_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply every decoder block in sequence."""
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


class DecoderOnlyTransformer(nnx.Module):
    """Predict next-token logits from token and position embeddings."""

    token_embedding: nnx.Embed
    position_embedding: nnx.Embed
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
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=embedding_dim,
            rngs=rngs,
        )
        self.position_embedding = nnx.Embed(
            num_embeddings=context_length,
            features=embedding_dim,
            rngs=rngs,
        )
        self.decoder_stack = Decoder(
            embedding_dim,
            hidden_dim,
            num_heads,
            num_decoder_blocks,
            rngs=rngs,
        )

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """Return next-token logits for a batch of token ids."""
        if input_ids.shape[-1] > self.position_embedding.embedding.shape[0]:
            raise ValueError("input sequence length exceeds context length")

        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.decoder_stack(x)
        return self.token_embedding.attend(x)
