"""Milestone 001: smoothed character-level bigram language model."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

from experiment_artifacts import write_loss_artifacts

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
LAPLACE_SMOOTHING = 1.0
SEED = 1337
SAMPLE_LEN = 200


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


def build_bigram_probs(encoded: jax.Array, vocab_size: int) -> jax.Array:
    bigram_counts = jnp.full((vocab_size, vocab_size), LAPLACE_SMOOTHING, dtype=jnp.float32)
    prev_tokens = encoded[:-1]
    next_tokens = encoded[1:]
    bigram_counts = bigram_counts.at[prev_tokens, next_tokens].add(1.0)
    return bigram_counts / bigram_counts.sum(axis=1, keepdims=True)


def sample_text(probs: jax.Array, chars: list[str], sample_len: int, rng: jax.Array) -> str:
    rng, sample_rng = jax.random.split(rng)
    sample_id = int(jax.random.randint(sample_rng, shape=(), minval=0, maxval=len(chars)).item())
    sample = [chars[sample_id]]
    for _ in range(sample_len - 1):
        rng, sample_rng = jax.random.split(rng)
        sample_id = int(jax.random.categorical(sample_rng, jnp.log(probs[sample_id])).item())
        sample.append(chars[sample_id])
    return "".join(sample)


def main() -> None:
    total_start = perf_counter()
    rng = set_seed(SEED)
    tokens = load_text(DATA_PATH)

    chars = sorted(set(tokens))
    char_to_id = {char: idx for idx, char in enumerate(chars)}
    vocab_size = len(char_to_id)

    encoded = jnp.asarray([char_to_id[ch] for ch in tokens], dtype=jnp.int32)
    probs = build_bigram_probs(encoded, vocab_size)
    prev_tokens = encoded[:-1]
    next_tokens = encoded[1:]
    cross_entropy = -jnp.log(probs[prev_tokens, next_tokens]).mean()
    sample = sample_text(probs, chars, SAMPLE_LEN, rng)
    loss_value = float(cross_entropy.item())
    loss_history = [(0, loss_value, loss_value)]
    loss_history_csv, loss_curve_svg = write_loss_artifacts(Path(__file__), loss_history)
    total_seconds = perf_counter() - total_start

    print(f"cross_entropy={loss_value:.6f}")
    print(f"loss_history_csv={loss_history_csv}")
    print(f"loss_curve_svg={loss_curve_svg}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
