from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from experiment_artifacts import write_loss_artifacts

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
SEQUENCE_LENGTH = 16
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.05
TRAIN_STEPS = 50_000
LOSS_EMA_DECAY = 0.95
LOG_INTERVAL = 1000


Model = dict[str, jax.Array]


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


def init_model(vocab_size: int, rng: jax.Array) -> Model:
    tanh_gain = 5.0 / 3.0
    embedding_rng, input_rng, recurrent_rng, output_rng = jax.random.split(rng, 4)
    return {
        "embedding_table": jax.random.normal(
            embedding_rng,
            (vocab_size, EMBEDDING_DIM),
            dtype=jnp.float32,
        )
        * 0.1,
        "input_weights": jax.random.normal(
            input_rng,
            (EMBEDDING_DIM, HIDDEN_DIM),
            dtype=jnp.float32,
        )
        * (tanh_gain / math.sqrt(EMBEDDING_DIM)),
        "recurrent_weights": jax.random.normal(
            recurrent_rng,
            (HIDDEN_DIM, HIDDEN_DIM),
            dtype=jnp.float32,
        )
        * (tanh_gain / math.sqrt(HIDDEN_DIM)),
        "hidden_bias": jnp.zeros((HIDDEN_DIM,), dtype=jnp.float32),
        "output_weights": jax.random.normal(
            output_rng,
            (HIDDEN_DIM, vocab_size),
            dtype=jnp.float32,
        )
        * (1.0 / math.sqrt(HIDDEN_DIM)),
        "output_bias": jnp.zeros((vocab_size,), dtype=jnp.float32),
    }


def init_hidden_state(batch_size: int) -> jax.Array:
    return jnp.zeros((batch_size, HIDDEN_DIM), dtype=jnp.float32)


def build_sequences(
    token_ids: jax.Array,
    start_positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    offsets = jnp.arange(SEQUENCE_LENGTH + 1, dtype=start_positions.dtype)
    sequence_token_ids = token_ids[start_positions[:, None] + offsets]
    return sequence_token_ids[:, :-1], sequence_token_ids[:, 1:]


def rnn_step(
    input_token_ids: jax.Array,
    previous_hidden_state: jax.Array,
    model: Model,
) -> tuple[jax.Array, jax.Array]:
    embedded_tokens = model["embedding_table"][input_token_ids]
    hidden_state = jnp.tanh(
        embedded_tokens @ model["input_weights"]
        + previous_hidden_state @ model["recurrent_weights"]
        + model["hidden_bias"]
    )
    logits = hidden_state @ model["output_weights"] + model["output_bias"]
    return logits, hidden_state


def forward_sequence(
    input_token_ids: jax.Array,
    model: Model,
    initial_hidden_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    batch_size = input_token_ids.shape[0]
    hidden_state = init_hidden_state(batch_size) if initial_hidden_state is None else initial_hidden_state

    def scan_step(
        current_hidden_state: jax.Array,
        step_input_token_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        step_logits, next_hidden_state = rnn_step(step_input_token_ids, current_hidden_state, model)
        return next_hidden_state, step_logits

    hidden_state, logits_by_step = jax.lax.scan(scan_step, hidden_state, input_token_ids.T)
    return jnp.swapaxes(logits_by_step, 0, 1), hidden_state


def sequence_loss(logits_by_step: jax.Array, target_token_ids: jax.Array) -> jax.Array:
    vocab_size = logits_by_step.shape[-1]
    logits = logits_by_step.reshape((-1, vocab_size))
    targets = target_token_ids.reshape((-1,))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(targets.shape[0]), targets])


def loss_fn(model: Model, input_token_ids: jax.Array, target_token_ids: jax.Array) -> jax.Array:
    logits_by_step, _ = forward_sequence(input_token_ids, model)
    return sequence_loss(logits_by_step, target_token_ids)


def train_step(
    model: Model,
    input_token_ids: jax.Array,
    target_token_ids: jax.Array,
) -> tuple[Model, jax.Array]:
    loss, grads = jax.value_and_grad(loss_fn)(model, input_token_ids, target_token_ids)
    model = jax.tree_util.tree_map(lambda param, grad: param - LEARNING_RATE * grad, model, grads)
    return model, loss


@partial(jax.jit, static_argnames=("num_steps",))
def train_steps(
    model: Model,
    token_ids: jax.Array,
    rng: jax.Array,
    num_steps: int,
) -> tuple[Model, jax.Array, jax.Array]:
    def scan_step(carry: tuple[Model, jax.Array], _: None) -> tuple[tuple[Model, jax.Array], jax.Array]:
        current_model, current_rng = carry
        current_rng, batch_rng = jax.random.split(current_rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(BATCH_SIZE,),
            minval=0,
            maxval=token_ids.shape[0] - SEQUENCE_LENGTH,
        )
        input_token_ids, target_token_ids = build_sequences(token_ids, start_positions)
        current_model, loss = train_step(current_model, input_token_ids, target_token_ids)
        return (current_model, current_rng), loss

    (model, rng), losses = jax.lax.scan(scan_step, (model, rng), length=num_steps)
    return model, rng, losses


@jax.jit
def evaluate_batch_loss(
    token_ids: jax.Array,
    start_positions: jax.Array,
    model: Model,
) -> jax.Array:
    input_token_ids, target_token_ids = build_sequences(token_ids, start_positions)
    return loss_fn(model, input_token_ids, target_token_ids)


def evaluate_split(token_ids: jax.Array, model: Model) -> float:
    start_positions = jnp.arange(0, token_ids.shape[0] - SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=jnp.int32)
    total_loss = 0.0
    total_sequences = 0

    for batch_start in range(0, start_positions.shape[0], EVAL_BATCH_SIZE):
        batch_positions = start_positions[batch_start : batch_start + EVAL_BATCH_SIZE]
        batch_loss = evaluate_batch_loss(token_ids, batch_positions, model)
        batch_sequence_count = int(batch_positions.shape[0])
        total_loss += float(batch_loss.item()) * batch_sequence_count
        total_sequences += batch_sequence_count

    return total_loss / total_sequences


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: Model,
    seed_token_ids: jax.Array,
    rng: jax.Array,
) -> str:
    if sample_length <= 0:
        return ""

    rng, seed_rng = jax.random.split(rng)
    seed_index = int(
        jax.random.randint(
            seed_rng,
            shape=(),
            minval=0,
            maxval=seed_token_ids.shape[0],
        ).item()
    )
    token_id = int(seed_token_ids[seed_index].item())
    sample = [vocab_chars[token_id]]
    current_token_ids = jnp.asarray([token_id], dtype=jnp.int32)
    hidden_state = init_hidden_state(1)

    for _ in range(sample_length - 1):
        logits, hidden_state = rnn_step(current_token_ids, hidden_state, model)
        rng, token_rng = jax.random.split(rng)
        token_id = int(jax.random.categorical(token_rng, logits[0]).item())
        sample.append(vocab_chars[token_id])
        current_token_ids = jnp.asarray([token_id], dtype=jnp.int32)

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
    if train_token_ids.shape[0] <= SEQUENCE_LENGTH or val_token_ids.shape[0] <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Dataset splits are too small for sequence length {SEQUENCE_LENGTH}. "
            "Need at least one full input sequence plus one target token in each split."
        )

    rng, model_rng = jax.random.split(rng)
    model = init_model(vocab_size, model_rng)
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
    sample = sample_text(vocab_chars, SAMPLE_LENGTH, model, train_token_ids, sample_rng)
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
