from pathlib import Path
import math

from flax import nnx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import optax  # pyright: ignore

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


class LayerNorm(nnx.Module):
    scale: nnx.Param[jax.Array]
    shift: nnx.Param[jax.Array]

    def __init__(self, features: int):
        self.scale = nnx.Param(jnp.ones((features,)))
        self.shift = nnx.Param(jnp.zeros((features,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + LAYER_NORM_EPS)
        return self.scale * normalized + self.shift


class Embedding(nnx.Module):
    weight: nnx.Param[jax.Array]

    def __init__(self, num_embeddings: int, embedding_dim: int, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(rngs.params.normal((num_embeddings, embedding_dim)) * 0.1)

    def __call__(self, indices: jax.Array) -> jax.Array:
        return self.weight[indices]


class Linear(nnx.Module):
    weight: nnx.Param[jax.Array]
    bias: nnx.Param[jax.Array] | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        bias: bool = True,
    ):
        scale = 1.0 / math.sqrt(in_features)
        self.weight = nnx.Param(rngs.params.normal((in_features, out_features)) * scale)
        self.bias = nnx.Param(jnp.zeros((out_features,))) if bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


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

    def __call__(self, x: jax.Array) -> jax.Array:
        attention_residual = x + self.attention(x)
        attention_block_output = self.attention_norm(attention_residual)

        feed_forward_residual = attention_block_output + self.feed_forward(attention_block_output)
        return self.feed_forward_norm(feed_forward_residual)


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

    def __call__(self, x: jax.Array) -> jax.Array:
        for block in self.blocks:
            x = block(x)
        return x


class LanguageModel(nnx.Module):
    token_embedding: Embedding
    position_embedding: Embedding
    decoder: Decoder
    lm_head: Linear

    def __init__(self, vocab_size: int, *, rngs: nnx.Rngs):
        self.token_embedding = Embedding(vocab_size, EMBEDDING_DIM, rngs=rngs)
        self.position_embedding = Embedding(CONTEXT_WINDOW, EMBEDDING_DIM, rngs=rngs)
        self.decoder = Decoder(EMBEDDING_DIM, HIDDEN_DIM, NUM_HEADS, NUM_DECODER_BLOCKS, rngs=rngs)
        self.lm_head = Linear(EMBEDDING_DIM, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        decoder_input = token_embeddings + position_embeddings
        decoder_output = self.decoder(decoder_input)
        return self.lm_head(decoder_output)


def loss_fn(model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@nnx.jit
def train_step(
    model: LanguageModel,
    optimizer: nnx.Optimizer[LanguageModel],
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    loss, grads = nnx.value_and_grad(loss_fn)(model, input_ids, target_ids)
    optimizer.update(model, grads)
    return loss


def sample_batch(batch_key: jax.Array, token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    max_start = token_ids.shape[0] - CONTEXT_WINDOW
    start_positions = jax.random.randint(batch_key, (BATCH_SIZE,), 0, max_start)
    input_positions = start_positions[:, None] + jnp.arange(CONTEXT_WINDOW)
    input_ids = token_ids[input_positions]
    target_ids = token_ids[input_positions + 1]
    return input_ids, target_ids


rngs = nnx.Rngs(SEED)
corpus = DATA_PATH.read_text(encoding="utf-8")
vocab_chars = sorted(set(corpus))
char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
token_ids = jnp.asarray([char_to_id[ch] for ch in corpus], dtype=jnp.int32)

model = LanguageModel(len(vocab_chars), rngs=rngs)
optimizer = nnx.Optimizer(model, optax.sgd(LEARNING_RATE), wrt=nnx.Param)

batch_rng = jax.random.key(SEED)
for step in range(TRAIN_STEPS):
    batch_rng, step_rng = jax.random.split(batch_rng)
    input_ids, target_ids = sample_batch(step_rng, token_ids)
    loss = train_step(model, optimizer, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
