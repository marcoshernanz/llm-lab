from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from lib import plotting
from lib.eval import evaluate_positions, evaluate_split, sample_evaluation_positions
from lib.plotting import LossTracker


def shifted_token_loss(_model: object, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    del _model
    return jnp.mean((target_ids - input_ids).astype(jnp.float32))


def test_sample_evaluation_positions_returns_unique_positions_in_range() -> None:
    tokens = jnp.arange(10, dtype=jnp.int32)
    positions = sample_evaluation_positions(
        tokens,
        context_length=3,
        subset_size=20,
        rng=jax.random.key(0),
    )

    assert positions.shape == (7,)
    assert jnp.all(positions >= 0)
    assert jnp.all(positions < 7)
    assert jnp.unique(positions).shape == positions.shape


def test_evaluate_positions_averages_loss_over_selected_positions() -> None:
    tokens = jnp.arange(10, dtype=jnp.int32)
    positions = jnp.array([0, 2, 4], dtype=jnp.int32)

    loss = evaluate_positions(
        tokens,
        positions,
        object(),
        shifted_token_loss,
        context_length=3,
        batch_size=2,
    )

    assert loss == pytest.approx(1.0)


def test_evaluate_split_averages_loss_over_full_split() -> None:
    tokens = jnp.arange(8, dtype=jnp.int32)

    loss = evaluate_split(
        tokens,
        object(),
        shifted_token_loss,
        context_length=3,
        batch_size=2,
    )

    assert loss == pytest.approx(1.0)


def test_loss_tracker_saves_validation_subset_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plotting, "ARTIFACTS_ROOT", tmp_path)

    tracker = LossTracker()
    tracker.log(step=10, train_loss=1.5, validation_subset_loss=1.8)
    tracker.log(step=20, train_loss=1.2, validation_subset_loss=1.6)

    csv_path, svg_path = tracker.save(script_path=Path("/tmp/example_experiment.py"))

    csv_text = csv_path.read_text(encoding="utf-8")
    svg_text = svg_path.read_text(encoding="utf-8")

    assert csv_path.parent.parent == tmp_path / "example_experiment"
    assert "validation_subset" in csv_text
    assert "validation subset" in svg_text
