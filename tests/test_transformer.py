import jax.numpy as jnp
import pytest
from flax import nnx

from models.layers import Embedding
from models.layers import LayerNorm
from models.layers import Linear
from models.transformer import CausalSelfAttention
from models.transformer import DecoderOnlyTransformer


def test_layer_norm_preserves_input_shape() -> None:
    layer_norm = LayerNorm(4)
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)

    output = layer_norm(x)

    assert output.shape == x.shape


def test_embedding_returns_embedding_vectors_for_indices() -> None:
    embedding = Embedding(10, 6, rngs=nnx.Rngs(0))
    indices = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)

    output = embedding(indices)

    assert output.shape == (2, 3, 6)


def test_linear_projects_last_dimension() -> None:
    linear = Linear(4, 7, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 3, 4), dtype=jnp.float32)

    output = linear(x)

    assert output.shape == (2, 3, 7)


def test_causal_self_attention_preserves_embedding_shape() -> None:
    attention = CausalSelfAttention(embedding_dim=8, num_heads=2, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 5, 8), dtype=jnp.float32)

    output = attention(x)

    assert output.shape == x.shape


def test_decoder_only_transformer_outputs_vocab_logits() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_decoder_blocks=2,
        rngs=nnx.Rngs(0),
    )
    input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)

    logits = model(input_ids)

    assert logits.shape == (2, 4, 32)
    assert jnp.isfinite(logits).all()


def test_decoder_only_transformer_rejects_inputs_longer_than_context_length() -> None:
    model = DecoderOnlyTransformer(
        vocab_size=32,
        context_length=4,
        embedding_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_decoder_blocks=2,
        rngs=nnx.Rngs(0),
    )
    input_ids = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)

    with pytest.raises(ValueError, match="input sequence length exceeds context length"):
        _ = model(input_ids)


def test_causal_self_attention_rejects_invalid_head_configuration() -> None:
    with pytest.raises(ValueError, match="embedding_dim must be divisible by num_heads"):
        _ = CausalSelfAttention(embedding_dim=10, num_heads=3, rngs=nnx.Rngs(0))
