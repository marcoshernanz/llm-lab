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
SEQUENCE_LENGTH = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 256
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.02
TRAIN_STEPS = 50_000
GRAD_CLIP_NORM = 1.0
LOSS_EMA_DECAY = 0.95
LOG_INTERVAL = 1000
GRU_GATE_COUNT = 3


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
    gate_dim = GRU_GATE_COUNT * HIDDEN_DIM
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
            (EMBEDDING_DIM, gate_dim),
            dtype=jnp.float32,
        )
        * (1.0 / math.sqrt(EMBEDDING_DIM)),
        "input_bias": jnp.zeros((gate_dim,), dtype=jnp.float32),
        "recurrent_weights": jax.random.normal(
            recurrent_rng,
            (HIDDEN_DIM, gate_dim),
            dtype=jnp.float32,
        )
        * (1.0 / math.sqrt(HIDDEN_DIM)),
        "recurrent_bias": jnp.zeros((gate_dim,), dtype=jnp.float32),
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


def build_streams(token_ids: jax.Array, batch_size: int) -> tuple[jax.Array, jax.Array]:
    usable_token_count = ((token_ids.shape[0] - 1) // batch_size) * batch_size
    if usable_token_count == 0:
        raise ValueError(
            f"Dataset split with {token_ids.shape[0]} tokens is too small for batch size {batch_size}."
        )

    input_streams = token_ids[:usable_token_count].reshape((batch_size, -1))
    target_streams = token_ids[1 : usable_token_count + 1].reshape((batch_size, -1))
    if input_streams.shape[1] < SEQUENCE_LENGTH:
        raise ValueError(
            f"Stream length {input_streams.shape[1]} is too short for sequence length "
            f"{SEQUENCE_LENGTH} at batch size {batch_size}."
        )
    return input_streams, target_streams


def get_sequence_chunk(
    input_streams: jax.Array,
    target_streams: jax.Array,
    chunk_start: jax.Array | int,
) -> tuple[jax.Array, jax.Array]:
    chunk_start = jnp.asarray(chunk_start, dtype=jnp.int32)
    input_chunk = jax.lax.dynamic_slice(
        input_streams,
        (0, chunk_start),
        (input_streams.shape[0], SEQUENCE_LENGTH),
    )
    target_chunk = jax.lax.dynamic_slice(
        target_streams,
        (0, chunk_start),
        (target_streams.shape[0], SEQUENCE_LENGTH),
    )
    return input_chunk, target_chunk


def split_gru_activations(
    activations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    reset, update, candidate = jnp.split(activations, GRU_GATE_COUNT, axis=1)
    return reset, update, candidate


def gru_step(
    input_token_ids: jax.Array,
    previous_hidden_state: jax.Array,
    model: Model,
) -> tuple[jax.Array, jax.Array]:
    embedded_tokens = model["embedding_table"][input_token_ids]
    input_activations = embedded_tokens @ model["input_weights"] + model["input_bias"]
    recurrent_activations = (
        previous_hidden_state @ model["recurrent_weights"] + model["recurrent_bias"]
    )
    input_reset, input_update, input_candidate = split_gru_activations(input_activations)
    recurrent_reset, recurrent_update, recurrent_candidate = split_gru_activations(
        recurrent_activations
    )

    reset_gate = jax.nn.sigmoid(input_reset + recurrent_reset)
    update_gate = jax.nn.sigmoid(input_update + recurrent_update)
    candidate_hidden_state = jnp.tanh(input_candidate + reset_gate * recurrent_candidate)
    hidden_state = (
        update_gate * previous_hidden_state + (1.0 - update_gate) * candidate_hidden_state
    )
    logits = hidden_state @ model["output_weights"] + model["output_bias"]
    return logits, hidden_state


def forward_sequence(
    input_token_ids: jax.Array,
    model: Model,
    initial_hidden_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    batch_size = input_token_ids.shape[0]
    hidden_state = (
        init_hidden_state(batch_size) if initial_hidden_state is None else initial_hidden_state
    )

    def scan_step(
        current_hidden_state: jax.Array,
        step_input_token_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        step_logits, next_hidden_state = gru_step(step_input_token_ids, current_hidden_state, model)
        return next_hidden_state, step_logits

    hidden_state, logits_by_step = jax.lax.scan(scan_step, hidden_state, input_token_ids.T)
    return jnp.swapaxes(logits_by_step, 0, 1), hidden_state


def sequence_loss(logits_by_step: jax.Array, target_token_ids: jax.Array) -> jax.Array:
    vocab_size = logits_by_step.shape[-1]
    logits = logits_by_step.reshape((-1, vocab_size))
    targets = target_token_ids.reshape((-1,))
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(targets.shape[0]), targets])


def loss_and_hidden(
    model: Model,
    input_token_ids: jax.Array,
    target_token_ids: jax.Array,
    initial_hidden_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits_by_step, hidden_state = forward_sequence(input_token_ids, model, initial_hidden_state)
    return sequence_loss(logits_by_step, target_token_ids), hidden_state


def global_norm(tree: Model) -> jax.Array:
    squared_norm = sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree))
    return jnp.sqrt(squared_norm)


def train_step(
    model: Model,
    input_token_ids: jax.Array,
    target_token_ids: jax.Array,
    initial_hidden_state: jax.Array,
) -> tuple[Model, jax.Array, jax.Array, jax.Array]:
    (loss, next_hidden_state), grads = jax.value_and_grad(loss_and_hidden, argnums=0, has_aux=True)(
        model,
        input_token_ids,
        target_token_ids,
        initial_hidden_state,
    )
    total_grad_norm = global_norm(grads)
    grad_scale = jnp.minimum(1.0, GRAD_CLIP_NORM / (total_grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda grad: grad * grad_scale, grads)
    model = jax.tree_util.tree_map(lambda param, grad: param - LEARNING_RATE * grad, model, grads)
    return model, loss, total_grad_norm, next_hidden_state


@partial(jax.jit, static_argnames=("num_steps",))
def train_steps(
    model: Model,
    input_streams: jax.Array,
    target_streams: jax.Array,
    hidden_state: jax.Array,
    chunk_start: jax.Array,
    num_steps: int,
) -> tuple[Model, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    stream_length = input_streams.shape[1]

    def scan_step(
        carry: tuple[Model, jax.Array, jax.Array],
        _: None,
    ) -> tuple[tuple[Model, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
        current_model, current_hidden_state, current_chunk_start = carry
        should_reset = current_chunk_start + SEQUENCE_LENGTH > stream_length
        current_hidden_state = jax.lax.select(
            should_reset,
            init_hidden_state(input_streams.shape[0]),
            current_hidden_state,
        )
        current_chunk_start = jnp.where(should_reset, 0, current_chunk_start)
        input_chunk, target_chunk = get_sequence_chunk(
            input_streams,
            target_streams,
            current_chunk_start,
        )
        current_model, loss, grad_norm, next_hidden_state = train_step(
            current_model,
            input_chunk,
            target_chunk,
            current_hidden_state,
        )
        next_hidden_state = jax.lax.stop_gradient(next_hidden_state)
        hidden_norm = jnp.linalg.norm(next_hidden_state, axis=1).mean()
        next_chunk_start = current_chunk_start + SEQUENCE_LENGTH
        return (
            current_model,
            next_hidden_state,
            next_chunk_start,
        ), (
            loss,
            grad_norm,
            hidden_norm,
        )

    (
        (model, hidden_state, chunk_start),
        (losses, grad_norms, hidden_norms),
    ) = jax.lax.scan(
        scan_step,
        (model, hidden_state, chunk_start),
        length=num_steps,
    )
    return model, hidden_state, chunk_start, losses, grad_norms, hidden_norms


@jax.jit
def evaluate_chunk(
    input_chunk: jax.Array,
    target_chunk: jax.Array,
    hidden_state: jax.Array,
    model: Model,
) -> tuple[jax.Array, jax.Array]:
    logits_by_step, next_hidden_state = forward_sequence(input_chunk, model, hidden_state)
    return sequence_loss(logits_by_step, target_chunk), next_hidden_state


def evaluate_split(token_ids: jax.Array, model: Model) -> float:
    input_streams, target_streams = build_streams(token_ids, EVAL_BATCH_SIZE)
    stream_length = input_streams.shape[1]
    hidden_state = init_hidden_state(EVAL_BATCH_SIZE)
    total_loss = 0.0
    total_tokens = 0

    for chunk_start in range(0, stream_length - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
        input_chunk, target_chunk = get_sequence_chunk(input_streams, target_streams, chunk_start)
        batch_loss, hidden_state = evaluate_chunk(input_chunk, target_chunk, hidden_state, model)
        batch_token_count = int(np.prod(input_chunk.shape))
        total_loss += float(batch_loss.item()) * batch_token_count
        total_tokens += batch_token_count

    return total_loss / total_tokens


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: Model,
    seed_token_ids: jax.Array,
    rng: jax.Array,
) -> str:
    if sample_length <= 0:
        return ""

    rng, primer_rng = jax.random.split(rng)
    primer_start = int(
        jax.random.randint(
            primer_rng,
            shape=(),
            minval=0,
            maxval=seed_token_ids.shape[0] - SEQUENCE_LENGTH + 1,
        ).item()
    )
    primer_token_ids = seed_token_ids[primer_start : primer_start + SEQUENCE_LENGTH]
    sample = [vocab_chars[int(token_id)] for token_id in primer_token_ids[:sample_length].tolist()]
    hidden_state = init_hidden_state(1)

    if SEQUENCE_LENGTH > 1:
        _, hidden_state = forward_sequence(primer_token_ids[:-1][None, :], model, hidden_state)

    current_token_ids = primer_token_ids[-1:]
    for _ in range(max(sample_length - len(sample), 0)):
        logits, hidden_state = gru_step(current_token_ids, hidden_state, model)
        rng, token_rng = jax.random.split(rng)
        next_token_id = int(jax.random.categorical(token_rng, logits[0]).item())
        sample.append(vocab_chars[next_token_id])
        current_token_ids = jnp.asarray([next_token_id], dtype=jnp.int32)

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

    train_input_streams, train_target_streams = build_streams(train_token_ids, BATCH_SIZE)
    rng, model_rng = jax.random.split(rng)
    model = init_model(vocab_size, model_rng)
    hidden_state = init_hidden_state(BATCH_SIZE)
    chunk_start = jnp.asarray(0, dtype=jnp.int32)
    loss_history: list[tuple[int, float, float]] = []
    ema_loss: float | None = None
    train_start = perf_counter()

    for step_group_start in range(0, TRAIN_STEPS, LOG_INTERVAL):
        group_steps = min(LOG_INTERVAL, TRAIN_STEPS - step_group_start)
        (
            model,
            hidden_state,
            chunk_start,
            losses,
            grad_norms,
            hidden_norms,
        ) = train_steps(
            model,
            train_input_streams,
            train_target_streams,
            hidden_state,
            chunk_start,
            group_steps,
        )
        losses_np = np.asarray(jax.device_get(losses), dtype=np.float32)
        grad_norms_np = np.asarray(jax.device_get(grad_norms), dtype=np.float32)
        hidden_norms_np = np.asarray(jax.device_get(hidden_norms), dtype=np.float32)

        for offset, raw_loss in enumerate(losses_np):
            step = step_group_start + offset
            raw_loss_value = float(raw_loss)
            ema_loss = (
                raw_loss_value
                if ema_loss is None
                else LOSS_EMA_DECAY * ema_loss + (1.0 - LOSS_EMA_DECAY) * raw_loss_value
            )
            loss_history.append((step, raw_loss_value, ema_loss))

        print(
            "step="
            f"{step_group_start} "
            f"loss={losses_np[0]:.6f} "
            f"ema_loss={loss_history[step_group_start][2]:.6f} "
            f"grad_norm={grad_norms_np[0]:.6f} "
            f"hidden_norm={hidden_norms_np[0]:.6f}"
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
