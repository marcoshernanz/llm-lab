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
BATCH_SIZE = 32
SAMPLE_LENGTH = 200
LEARNING_RATE = 0.05
TRAIN_STEPS = 50_000
CONTEXT_LENGTH = 4
LOSS_EMA_DECAY = 0.95
LOG_INTERVAL = 1000


Model = dict[str, jax.Array]


def set_seed(seed: int) -> jax.Array:
    return jax.random.key(seed)


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


def init_model(vocab_size: int, rng: jax.Array) -> Model:
    input_dim = EMBEDDING_DIM * CONTEXT_LENGTH
    embedding_rng, output_rng = jax.random.split(rng)
    return {
        "embedding_table": jax.random.normal(
            embedding_rng,
            (vocab_size, EMBEDDING_DIM),
            dtype=jnp.float32,
        )
        * 0.1,
        "output_weights": jax.random.normal(
            output_rng,
            (input_dim, vocab_size),
            dtype=jnp.float32,
        )
        * (1.0 / math.sqrt(input_dim)),
        "output_bias": jnp.zeros((vocab_size,), dtype=jnp.float32),
    }


def build_examples(
    token_ids: jax.Array,
    start_positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    offsets = jnp.arange(CONTEXT_LENGTH, dtype=start_positions.dtype)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions + CONTEXT_LENGTH]
    return input_ids, target_ids


def forward(input_ids: jax.Array, model: Model) -> jax.Array:
    embedded = model["embedding_table"][input_ids].reshape((input_ids.shape[0], -1))
    return embedded @ model["output_weights"] + model["output_bias"]


def loss_fn(model: Model, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = forward(input_ids, model)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(target_ids.shape[0]), target_ids])


def train_step(model: Model, input_ids: jax.Array, target_ids: jax.Array) -> tuple[Model, jax.Array]:
    loss, grads = jax.value_and_grad(loss_fn)(model, input_ids, target_ids)
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
            maxval=token_ids.shape[0] - CONTEXT_LENGTH,
        )
        input_ids, target_ids = build_examples(token_ids, start_positions)
        current_model, loss = train_step(current_model, input_ids, target_ids)
        return (current_model, current_rng), loss

    (model, rng), losses = jax.lax.scan(scan_step, (model, rng), length=num_steps)
    return model, rng, losses


@jax.jit
def evaluate_loss(token_ids: jax.Array, model: Model) -> jax.Array:
    start_positions = jnp.arange(token_ids.shape[0] - CONTEXT_LENGTH, dtype=jnp.int32)
    input_ids, target_ids = build_examples(token_ids, start_positions)
    return loss_fn(model, input_ids, target_ids)


def evaluate_split(token_ids: jax.Array, model: Model) -> float:
    return float(evaluate_loss(token_ids, model).item())


def sample_text(
    vocab_chars: list[str],
    sample_length: int,
    model: Model,
    seed_token_ids: jax.Array,
    rng: jax.Array,
) -> str:
    rng, seed_rng = jax.random.split(rng)
    seed_start = int(
        jax.random.randint(
            seed_rng,
            shape=(),
            minval=0,
            maxval=seed_token_ids.shape[0] - CONTEXT_LENGTH + 1,
        ).item()
    )
    context = seed_token_ids[seed_start : seed_start + CONTEXT_LENGTH]
    sample = [vocab_chars[int(token_id)] for token_id in context[:sample_length].tolist()]

    for _ in range(max(sample_length - len(sample), 0)):
        logits = forward(context[None, :], model)
        rng, token_rng = jax.random.split(rng)
        next_token_id = int(jax.random.categorical(token_rng, logits[0]).item())
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
