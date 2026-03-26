from __future__ import annotations

from pathlib import Path
import math
from time import perf_counter

from flax import nnx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import optax  # pyright: ignore

from experiment_artifacts import write_loss_artifacts

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
NUM_DECODER_BLOCKS = 4
HIDDEN_DIM = 128
CONTEXT_LENGTH = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.02
TRAIN_STEPS = 50_000
LOSS_EMA_DECAY = 0.95
LOG_INTERVAL = 1000
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
        self.position_embedding = Embedding(CONTEXT_LENGTH, EMBEDDING_DIM, rngs=rngs)
        self.decoder = Decoder(EMBEDDING_DIM, HIDDEN_DIM, NUM_HEADS, NUM_DECODER_BLOCKS, rngs=rngs)
        self.lm_head = Linear(EMBEDDING_DIM, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        decoder_input = token_embeddings + position_embeddings
        decoder_output = self.decoder(decoder_input)
        return self.lm_head(decoder_output)


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place tinyshakespeare.txt there before running this script."
        )
    text = path.read_text(encoding="utf-8")
    if len(text) < 2:
        raise ValueError("Dataset is too small. Need at least 2 characters.")
    return text


def build_examples(
    token_ids: jax.Array,
    start_positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    offsets = jnp.arange(CONTEXT_LENGTH, dtype=start_positions.dtype)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions[:, None] + offsets + 1]
    return input_ids, target_ids


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


def train_steps(
    model: LanguageModel,
    optimizer: nnx.Optimizer[LanguageModel],
    token_ids: jax.Array,
    rng: jax.Array,
    num_steps: int,
) -> tuple[jax.Array, list[float]]:
    losses: list[float] = []
    current_rng = rng

    for _ in range(num_steps):
        current_rng, batch_rng = jax.random.split(current_rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(BATCH_SIZE,),
            minval=0,
            maxval=token_ids.shape[0] - CONTEXT_LENGTH,
        )
        input_ids, target_ids = build_examples(token_ids, start_positions)
        loss = train_step(model, optimizer, input_ids, target_ids)
        losses.append(float(loss))

    return current_rng, losses


@nnx.jit
def evaluate_batch_loss(
    model: LanguageModel,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    return loss_fn(model, input_ids, target_ids)


def evaluate_split(token_ids: jax.Array, model: LanguageModel) -> float:
    max_start = token_ids.shape[0] - CONTEXT_LENGTH
    if max_start <= 0:
        raise ValueError(
            f"Dataset split is too small for context length {CONTEXT_LENGTH}. "
            "Need at least one full context window plus one target token."
        )

    total_loss = 0.0
    total_examples = 0

    for batch_start in range(0, max_start, EVAL_BATCH_SIZE):
        batch_end = min(batch_start + EVAL_BATCH_SIZE, max_start)
        start_positions = jnp.arange(batch_start, batch_end, dtype=jnp.int32)
        input_ids, target_ids = build_examples(token_ids, start_positions)
        batch_loss = evaluate_batch_loss(model, input_ids, target_ids)
        batch_size = int(start_positions.shape[0])
        total_loss += float(batch_loss) * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def generate_text(
    model: LanguageModel,
    vocab_chars: list[str],
    seed_token_ids: jax.Array,
    sample_length: int,
    rng: jax.Array,
) -> str:
    if sample_length <= 0:
        return ""

    rng, seed_rng = jax.random.split(rng)
    seed_start = int(
        jax.random.randint(
            seed_rng,
            shape=(),
            minval=0,
            maxval=seed_token_ids.shape[0] - CONTEXT_LENGTH,
        )
    )
    context = seed_token_ids[seed_start : seed_start + CONTEXT_LENGTH]
    sample = [vocab_chars[int(token_id)] for token_id in context[:sample_length].tolist()]

    for _ in range(max(sample_length - len(sample), 0)):
        logits = model(context[None, :])
        rng, token_rng = jax.random.split(rng)
        next_token_id = int(jax.random.categorical(token_rng, logits[0, -1]))
        sample.append(vocab_chars[next_token_id])
        context = jnp.concatenate((context[1:], jnp.asarray([next_token_id], dtype=jnp.int32)))

    return "".join(sample)


def main() -> None:
    total_start = perf_counter()
    rngs = nnx.Rngs(SEED)
    text = load_text(DATA_PATH)

    vocab_chars = sorted(set(text))
    char_to_index = {char: idx for idx, char in enumerate(vocab_chars)}
    vocab_size = len(char_to_index)

    token_ids = jnp.asarray([char_to_index[ch] for ch in text], dtype=jnp.int32)
    num_tokens = token_ids.shape[0]
    train_token_ids = token_ids[: int(num_tokens * 0.8)]
    val_token_ids = token_ids[int(num_tokens * 0.8) :]
    if train_token_ids.shape[0] <= CONTEXT_LENGTH or val_token_ids.shape[0] <= CONTEXT_LENGTH:
        raise ValueError(
            f"Dataset splits are too small for context length {CONTEXT_LENGTH}. "
            "Need at least one full context window plus one target token in each split."
        )

    model = LanguageModel(vocab_size, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.sgd(LEARNING_RATE), wrt=nnx.Param)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    train_start = perf_counter()

    batch_rng = jax.random.key(SEED)
    for chunk_start in range(0, TRAIN_STEPS, LOG_INTERVAL):
        chunk_steps = min(LOG_INTERVAL, TRAIN_STEPS - chunk_start)
        batch_rng, losses = train_steps(model, optimizer, train_token_ids, batch_rng, chunk_steps)
        losses_np = np.asarray(losses, dtype=np.float32)

        for offset, raw_loss in enumerate(losses_np):
            step = chunk_start + offset
            raw_loss_value = float(raw_loss)
            ema_loss = (
                raw_loss_value
                if ema_loss is None
                else LOSS_EMA_DECAY * ema_loss + (1.0 - LOSS_EMA_DECAY) * raw_loss_value
            )
            loss_history.append((step, raw_loss_value, ema_loss))

        print(
            f"step={chunk_start} loss={losses_np[0]:.6f} "
            f"ema_loss={loss_history[chunk_start][2]:.6f}"
        )

    train_seconds = perf_counter() - train_start
    train_loss = evaluate_split(train_token_ids, model)
    validation_loss = evaluate_split(val_token_ids, model)
    batch_rng, sample_rng = jax.random.split(batch_rng)
    sample = generate_text(model, vocab_chars, train_token_ids, SAMPLE_LENGTH, sample_rng)
    loss_history_csv, loss_curve_svg = write_loss_artifacts(Path(__file__), loss_history)
    total_seconds = perf_counter() - total_start

    print(f"train_loss={train_loss:.6f}")
    print(f"validation_loss={validation_loss:.6f}")
    print(f"loss_history_csv={loss_history_csv}")
    print(f"loss_curve_svg={loss_curve_svg}")
    print(f"train_seconds={train_seconds:.3f}")
    print(f"steps_per_second={TRAIN_STEPS / train_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
