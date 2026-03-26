from pathlib import Path
import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
BATCH_SIZE = 64
EMBEDDING_DIM = 128
NUM_HEADS = 4
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
NUM_DECODER_BLOCKS = 4
HIDDEN_DIM = 256
CONTEXT_WINDOW = 512
LEARNING_RATE = 0.05
TRAIN_STEPS = 5_000
LAYER_NORM_EPS = 1e-5


class LayerNorm(eqx.Module):
    scale: jax.Array
    shift: jax.Array

    def __init__(self):
        self.scale = jnp.ones((EMBEDDING_DIM,))
        self.shift = jnp.zeros((EMBEDDING_DIM,))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + LAYER_NORM_EPS)
        return self.scale * normalized + self.shift


class Embedding(eqx.Module):
    weight: jax.Array

    def __init__(self, num_embeddings: int, embedding_dim: int, rng: jax.Array):
        self.weight = jax.random.normal(rng, (num_embeddings, embedding_dim)) * 0.1

    def __call__(self, indices: jax.Array) -> jax.Array:
        return self.weight[indices]


class Linear(eqx.Module):
    weight: jax.Array
    bias: Optional[jax.Array]

    def __init__(self, in_features: int, out_features: int, rng: jax.Array, bias: bool = True):
        self.weight = jax.random.normal(rng, (in_features, out_features)) * (
            1.0 / math.sqrt(in_features)
        )
        self.bias = jnp.zeros((out_features,)) if bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


class CausalSelfAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear

    def __init__(self, rng: jax.Array):
        query_rng, key_rng, value_rng, output_rng = jax.random.split(rng, 4)
        self.query = Linear(EMBEDDING_DIM, NUM_HEADS * HEAD_DIM, query_rng, bias=False)
        self.key = Linear(EMBEDDING_DIM, NUM_HEADS * HEAD_DIM, key_rng, bias=False)
        self.value = Linear(EMBEDDING_DIM, NUM_HEADS * HEAD_DIM, value_rng, bias=False)
        self.output = Linear(NUM_HEADS * HEAD_DIM, EMBEDDING_DIM, output_rng, bias=False)

    def split_heads(self, x: jax.Array) -> jax.Array:
        batch_size, sequence_length, _ = x.shape
        head_shape = (batch_size, sequence_length, NUM_HEADS, HEAD_DIM)
        return x.reshape(head_shape).swapaxes(1, 2)

    def combine_heads(self, x: jax.Array) -> jax.Array:
        batch_size, _, sequence_length, _ = x.shape
        combined_shape = (batch_size, sequence_length, NUM_HEADS * HEAD_DIM)
        return x.swapaxes(1, 2).reshape(combined_shape)

    def __call__(self, x: jax.Array) -> jax.Array:
        sequence_length = x.shape[1]
        queries = self.split_heads(self.query(x))
        keys = self.split_heads(self.key(x))
        values = self.split_heads(self.value(x))

        attention_scores = (queries @ keys.mT) / math.sqrt(HEAD_DIM)
        causal_mask = jnp.triu(jnp.ones((sequence_length, sequence_length), dtype=bool), k=1)
        masked_attention_scores = jnp.where(causal_mask, -jnp.inf, attention_scores)
        attention_weights = jnn.softmax(masked_attention_scores, axis=-1)
        attended_values = attention_weights @ values
        combined_heads = self.combine_heads(attended_values)
        return self.output(combined_heads)


class FeedForward(eqx.Module):
    hidden: Linear
    output: Linear

    def __init__(self, rng: jax.Array):
        hidden_rng, output_rng = jax.random.split(rng, 2)
        self.hidden = Linear(EMBEDDING_DIM, HIDDEN_DIM, hidden_rng)
        self.output = Linear(HIDDEN_DIM, EMBEDDING_DIM, output_rng)

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden_activation = jnp.tanh(self.hidden(x))
        return self.output(hidden_activation)


class DecoderBlock(eqx.Module):
    attention: CausalSelfAttention
    attention_norm: LayerNorm
    feed_forward: FeedForward
    feed_forward_norm: LayerNorm

    def __init__(self, rng: jax.Array):
        attention_rng, feed_forward_rng = jax.random.split(rng, 2)
        self.attention = CausalSelfAttention(attention_rng)
        self.attention_norm = LayerNorm()
        self.feed_forward = FeedForward(feed_forward_rng)
        self.feed_forward_norm = LayerNorm()

    def __call__(self, x: jax.Array) -> jax.Array:
        attention_residual = x + self.attention(x)
        attention_block_output = self.attention_norm(attention_residual)

        feed_forward_residual = attention_block_output + self.feed_forward(attention_block_output)
        return self.feed_forward_norm(feed_forward_residual)


class Decoder(eqx.Module):
    blocks: tuple[DecoderBlock, ...]

    def __init__(self, rng: jax.Array):
        block_rngs = jax.random.split(rng, NUM_DECODER_BLOCKS)
        self.blocks = tuple(DecoderBlock(block_rng) for block_rng in block_rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for block in self.blocks:
            x = block(x)
        return x


class LanguageModel(eqx.Module):
    token_embedding: Embedding
    position_embedding: Embedding
    decoder: Decoder
    lm_head: Linear

    def __init__(self, rng: jax.Array, vocab_size: int):
        embedding_rng, position_rng, transformer_rng, logits_rng = jax.random.split(rng, 4)

        self.token_embedding = Embedding(vocab_size, EMBEDDING_DIM, embedding_rng)
        self.position_embedding = Embedding(CONTEXT_WINDOW, EMBEDDING_DIM, position_rng)
        self.decoder = Decoder(transformer_rng)
        self.lm_head = Linear(EMBEDDING_DIM, vocab_size, logits_rng)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        decoder_input = token_embeddings + position_embeddings
        decoder_output = self.decoder(decoder_input)
        return self.lm_head(decoder_output)


@eqx.filter_value_and_grad
def loss_fn(model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@eqx.filter_jit
def train_step(
    model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array
) -> tuple[LanguageModel, jax.Array]:
    loss, grads = loss_fn(model, input_ids, target_ids)
    updates = jax.tree_util.tree_map(lambda grad: -LEARNING_RATE * grad, grads)
    model = eqx.apply_updates(model, updates)
    return model, loss


def sample_batch(batch_key: jax.Array, token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    max_start = token_ids.shape[0] - CONTEXT_WINDOW
    start_positions = jax.random.randint(batch_key, (BATCH_SIZE,), 0, max_start)
    input_positions = start_positions[:, None] + jnp.arange(CONTEXT_WINDOW)
    input_ids = token_ids[input_positions]
    target_ids = token_ids[input_positions + 1]
    return input_ids, target_ids


key = jax.random.key(SEED)
corpus = DATA_PATH.read_text(encoding="utf-8")
vocab_chars = sorted(set(corpus))
char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
token_ids = jnp.asarray([char_to_id[ch] for ch in corpus], dtype=jnp.int32)

key, model_rng = jax.random.split(key)
model = LanguageModel(model_rng, len(vocab_chars))

for step in range(TRAIN_STEPS):
    key, batch_key = jax.random.split(key)
    input_ids, target_ids = sample_batch(batch_key, token_ids)
    model, loss = train_step(model, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
