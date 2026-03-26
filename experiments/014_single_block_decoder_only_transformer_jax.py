from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

from experiment_artifacts import write_loss_artifacts

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
ATTENTION_DIM = 32
CONTEXT_LENGTH = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.02
TRAIN_STEPS = 100_000
LOSS_EMA_DECAY = 0.95
LOG_INTERVAL = 1000
LAYER_NORM_EPS = 1e-5


class LayerNorm(eqx.Module):
    scale: jax.Array
    shift: jax.Array

    def __init__(self) -> None:
        self.scale = jnp.ones((EMBEDDING_DIM,), dtype=jnp.float32)
        self.shift = jnp.zeros((EMBEDDING_DIM,), dtype=jnp.float32)

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(variance + LAYER_NORM_EPS)
        return self.scale * normalized + self.shift


class Embedding(eqx.Module):
    weight: jax.Array

    def __init__(self, num_embeddings: int, embedding_dim: int, rng: jax.Array) -> None:
        self.weight = (
            jax.random.normal(rng, (num_embeddings, embedding_dim), dtype=jnp.float32) * 0.1
        )

    def __call__(self, indices: jax.Array) -> jax.Array:
        return self.weight[indices]


class Linear(eqx.Module):
    weight: jax.Array
    bias: Optional[jax.Array]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: jax.Array,
        bias: bool = True,
    ) -> None:
        self.weight = jax.random.normal(rng, (in_features, out_features), dtype=jnp.float32) * (
            1.0 / math.sqrt(in_features)
        )
        self.bias = jnp.zeros((out_features,), dtype=jnp.float32) if bias else None

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

    def __init__(self, rng: jax.Array) -> None:
        query_rng, key_rng, value_rng, output_rng = jax.random.split(rng, 4)
        self.query = Linear(EMBEDDING_DIM, ATTENTION_DIM, query_rng, bias=False)
        self.key = Linear(EMBEDDING_DIM, ATTENTION_DIM, key_rng, bias=False)
        self.value = Linear(EMBEDDING_DIM, ATTENTION_DIM, value_rng, bias=False)
        self.output = Linear(ATTENTION_DIM, EMBEDDING_DIM, output_rng, bias=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = (queries @ jnp.swapaxes(keys, -1, -2)) / math.sqrt(ATTENTION_DIM)
        causal_mask = jnp.triu(jnp.ones((x.shape[-2], x.shape[-2]), dtype=bool), k=1)
        masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
        attention_weights = jnn.softmax(masked_scores, axis=-1)
        mixed_values = attention_weights @ values
        return self.output(mixed_values)


class FeedForward(eqx.Module):
    hidden: Linear
    output: Linear

    def __init__(self, rng: jax.Array) -> None:
        hidden_rng, output_rng = jax.random.split(rng, 2)
        self.hidden = Linear(EMBEDDING_DIM, HIDDEN_DIM, hidden_rng)
        self.output = Linear(HIDDEN_DIM, EMBEDDING_DIM, output_rng)

    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = jnp.tanh(self.hidden(x))
        return self.output(hidden)


class DecoderBlock(eqx.Module):
    attention: CausalSelfAttention
    attention_norm: LayerNorm
    feed_forward: FeedForward
    feed_forward_norm: LayerNorm

    def __init__(self, rng: jax.Array) -> None:
        attention_rng, feed_forward_rng = jax.random.split(rng, 2)
        self.attention = CausalSelfAttention(attention_rng)
        self.attention_norm = LayerNorm()
        self.feed_forward = FeedForward(feed_forward_rng)
        self.feed_forward_norm = LayerNorm()

    def __call__(self, x: jax.Array) -> jax.Array:
        attention_block_output = self.attention_norm(x + self.attention(x))
        return self.feed_forward_norm(
            attention_block_output + self.feed_forward(attention_block_output)
        )


class LanguageModel(eqx.Module):
    token_embedding: Embedding
    position_embedding: Embedding
    decoder_block: DecoderBlock
    lm_head: Linear

    def __init__(self, rng: jax.Array, vocab_size: int) -> None:
        embedding_rng, position_rng, decoder_rng, lm_head_rng = jax.random.split(rng, 4)
        self.token_embedding = Embedding(vocab_size, EMBEDDING_DIM, embedding_rng)
        self.position_embedding = Embedding(CONTEXT_LENGTH, EMBEDDING_DIM, position_rng)
        self.decoder_block = DecoderBlock(decoder_rng)
        self.lm_head = Linear(EMBEDDING_DIM, vocab_size, lm_head_rng)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        decoder_input = token_embeddings + position_embeddings
        decoder_output = self.decoder_block(decoder_input)
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


def set_seed(seed: int) -> jax.Array:
    return jax.random.key(seed)


def build_examples(
    token_ids: jax.Array,
    start_positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    offsets = jnp.arange(CONTEXT_LENGTH, dtype=start_positions.dtype)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions[:, None] + offsets + 1]
    return input_ids, target_ids


@eqx.filter_value_and_grad
def loss_fn(model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@eqx.filter_jit
def train_step(
    model: LanguageModel,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> tuple[LanguageModel, jax.Array]:
    loss, grads = loss_fn(model, input_ids, target_ids)
    updates = jax.tree_util.tree_map(lambda grad: -LEARNING_RATE * grad, grads)
    model = eqx.apply_updates(model, updates)
    return model, loss


@eqx.filter_jit
def train_steps(
    model: LanguageModel,
    token_ids: jax.Array,
    rng: jax.Array,
    num_steps: int,
) -> tuple[LanguageModel, jax.Array, jax.Array]:
    def scan_step(
        carry: tuple[LanguageModel, jax.Array],
        _: None,
    ) -> tuple[tuple[LanguageModel, jax.Array], jax.Array]:
        current_model, current_rng = carry
        current_rng, batch_rng = jax.random.split(current_rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(BATCH_SIZE,),
            minval=0,
            maxval=token_ids.shape[0] - CONTEXT_LENGTH,
        )
        input_ids, target_ids = build_examples(token_ids, start_positions)
        current_model, loss = train_step(current_model, input_ids, target_ids)
        return (current_model, current_rng), loss

    (model, rng), losses = jax.lax.scan(scan_step, (model, rng), length=num_steps)
    return model, rng, losses


@eqx.filter_jit
def evaluate_batch_loss(
    model: LanguageModel,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


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
        total_loss += float(batch_loss.item()) * batch_size
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
        ).item()
    )
    context = seed_token_ids[seed_start : seed_start + CONTEXT_LENGTH]
    sample = [vocab_chars[int(token_id)] for token_id in context[:sample_length].tolist()]

    for _ in range(max(sample_length - len(sample), 0)):
        logits = model(context[None, :])
        rng, token_rng = jax.random.split(rng)
        next_token_id = int(jax.random.categorical(token_rng, logits[0, -1]).item())
        sample.append(vocab_chars[next_token_id])
        context = jnp.concatenate((context[1:], jnp.asarray([next_token_id], dtype=jnp.int32)))

    return "".join(sample)


def main() -> None:
    total_start = perf_counter()
    rng = set_seed(SEED)
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

    rng, model_rng = jax.random.split(rng)
    model = LanguageModel(model_rng, vocab_size)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    train_start = perf_counter()

    for chunk_start in range(0, TRAIN_STEPS, LOG_INTERVAL):
        chunk_steps = min(LOG_INTERVAL, TRAIN_STEPS - chunk_start)
        model, rng, losses = train_steps(model, train_token_ids, rng, chunk_steps)
        losses_np = np.asarray(jax.device_get(losses), dtype=np.float32)

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
    rng, sample_rng = jax.random.split(rng)
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
