from __future__ import annotations

import math
from typing import Any, TypeAlias
from functools import partial
from pathlib import Path
from time import perf_counter

from flax.core import FrozenDict
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiment_artifacts import write_loss_artifacts

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_DIM = 128
ATTENTION_DIM = 64
CONTEXT_LENGTH = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.02
TRAIN_STEPS = 100_000
LOSS_EMA_DECAY = 0.95
LOG_INTERVAL = 1000

Params: TypeAlias = FrozenDict[str, Any]


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


class SingleHeadAttentionLanguageModel(nn.Module):
    vocab_size: int
    embedding_dim: int = EMBEDDING_DIM
    attention_dim: int = ATTENTION_DIM
    context_length: int = CONTEXT_LENGTH

    def setup(self) -> None:
        embedding_init = nn.initializers.normal(stddev=0.1)
        input_projection_init = nn.initializers.normal(stddev=1.0 / math.sqrt(self.embedding_dim))
        output_projection_init = nn.initializers.normal(stddev=1.0 / math.sqrt(self.attention_dim))

        self.token_embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            embedding_init=embedding_init,
        )
        self.position_embedding = nn.Embed(
            num_embeddings=self.context_length,
            features=self.embedding_dim,
            embedding_init=embedding_init,
        )
        self.query = nn.Dense(
            features=self.attention_dim,
            use_bias=False,
            kernel_init=input_projection_init,
        )
        self.key = nn.Dense(
            features=self.attention_dim,
            use_bias=False,
            kernel_init=input_projection_init,
        )
        self.value = nn.Dense(
            features=self.attention_dim,
            use_bias=False,
            kernel_init=input_projection_init,
        )
        self.output = nn.Dense(
            features=self.embedding_dim,
            use_bias=False,
            kernel_init=output_projection_init,
        )
        self.lm_head = nn.Dense(
            features=self.vocab_size,
            kernel_init=input_projection_init,
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        _, sequence_length = input_ids.shape
        if sequence_length != self.context_length:
            raise ValueError(
                f"Input sequence length {sequence_length} does not match "
                f"context length {self.context_length}."
            )

        positions = jnp.arange(self.context_length, dtype=input_ids.dtype)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)[None, :, :]
        input_embeddings = token_embeddings + position_embeddings

        queries = self.query(input_embeddings)
        keys = self.key(input_embeddings)
        values = self.value(input_embeddings)

        scores = (queries @ jnp.swapaxes(keys, -1, -2)) / math.sqrt(self.attention_dim)
        causal_mask = jnp.triu(
            jnp.ones((self.context_length, self.context_length), dtype=bool),
            k=1,
        )
        masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
        attention_weights = nn.softmax(masked_scores, axis=-1)
        attention_output = attention_weights @ values
        output = self.output(attention_output)
        return self.lm_head(output)


def create_train_state(
    model: SingleHeadAttentionLanguageModel,
    rng: jax.Array,
) -> train_state.TrainState:
    dummy_input_ids = jnp.zeros((1, CONTEXT_LENGTH), dtype=jnp.int32)
    params = model.init(rng, dummy_input_ids)["params"]
    tx = optax.sgd(learning_rate=LEARNING_RATE)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def loss_fn(
    params: Params,
    state: train_state.TrainState,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    logits = state.apply_fn({"params": params}, input_ids)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@jax.jit
def train_step(
    state: train_state.TrainState,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> tuple[train_state.TrainState, jax.Array]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state, input_ids, target_ids)
    state = state.apply_gradients(grads=grads)
    return state, loss


@partial(jax.jit, static_argnames=("num_steps",))
def train_steps(
    state: train_state.TrainState,
    token_ids: jax.Array,
    rng: jax.Array,
    num_steps: int,
) -> tuple[train_state.TrainState, jax.Array, jax.Array]:
    def scan_step(
        carry: tuple[train_state.TrainState, jax.Array],
        _: None,
    ) -> tuple[tuple[train_state.TrainState, jax.Array], jax.Array]:
        current_state, current_rng = carry
        current_rng, batch_rng = jax.random.split(current_rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(BATCH_SIZE,),
            minval=0,
            maxval=token_ids.shape[0] - CONTEXT_LENGTH,
        )
        input_ids, target_ids = build_examples(token_ids, start_positions)
        current_state, loss = train_step(current_state, input_ids, target_ids)
        return (current_state, current_rng), loss

    (state, rng), losses = jax.lax.scan(scan_step, (state, rng), length=num_steps)
    return state, rng, losses


@jax.jit
def evaluate_batch_loss(
    state: train_state.TrainState,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    return loss_fn(state.params, state, input_ids, target_ids)


def evaluate_split(token_ids: jax.Array, state: train_state.TrainState) -> float:
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
        batch_loss = evaluate_batch_loss(state, input_ids, target_ids)
        batch_size = int(start_positions.shape[0])
        total_loss += float(batch_loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    state: train_state.TrainState,
    seed_token_ids: jax.Array,
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
        logits = state.apply_fn({"params": state.params}, context[None, :])
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
    model = SingleHeadAttentionLanguageModel(vocab_size=vocab_size)
    state = create_train_state(model, model_rng)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    train_start = perf_counter()

    for chunk_start in range(0, TRAIN_STEPS, LOG_INTERVAL):
        chunk_steps = min(LOG_INTERVAL, TRAIN_STEPS - chunk_start)
        state, rng, losses = train_steps(state, train_token_ids, rng, chunk_steps)
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
    train_loss = evaluate_split(train_token_ids, state)
    validation_loss = evaluate_split(val_token_ids, state)
    rng, sample_rng = jax.random.split(rng)
    sample = sample_text(vocab_chars, SAMPLE_LENGTH, state, train_token_ids, sample_rng)
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
