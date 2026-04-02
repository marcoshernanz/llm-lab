"""Save self-describing artifacts for reusable experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import jax

from lib.plotting import LossTracker


class RunArtifacts(dict[str, Path]):
    """Name the files produced for one saved experiment run."""


def resolve_execution_target(execution_target: str | None) -> str:
    """Choose a readable execution target when none is provided."""
    if execution_target:
        return execution_target
    return f"{jax.default_backend()} x{jax.device_count()}"


def build_run_metadata(
    *,
    script_path: Path,
    config: Mapping[str, object],
    execution_target: str | None,
    run_details: dict[str, object] | None = None,
    run_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assemble metadata so one artifact directory can explain one run."""
    metadata: dict[str, object] = {
        "script_name": script_path.name,
        "execution_target": resolve_execution_target(execution_target),
        "jax_backend": jax.default_backend(),
        "jax_device_count": jax.device_count(),
        **serialize_mapping(dict(config)),
    }

    if run_details is not None:
        metadata.update(serialize_mapping(run_details))
    if run_metrics is not None:
        metadata.update(serialize_mapping(run_metrics))

    train_seconds = metadata.get("train_seconds")
    train_steps = metadata.get("train_steps")
    batch_size = metadata.get("batch_size")
    context_length = metadata.get("context_length")
    tokens_per_step = None
    if isinstance(batch_size, (int, float)) and isinstance(context_length, (int, float)):
        tokens_per_step = batch_size * context_length
        metadata["tokens_per_step"] = tokens_per_step

    if isinstance(train_steps, (int, float)):
        if tokens_per_step is not None:
            metadata["train_tokens_seen"] = train_steps * tokens_per_step

    if isinstance(train_seconds, (int, float)) and train_seconds > 0:
        if isinstance(train_steps, (int, float)):
            steps_per_second = train_steps / train_seconds
            metadata["steps_per_second"] = steps_per_second
            if tokens_per_step is not None:
                metadata["tokens_per_second"] = steps_per_second * tokens_per_step

    return metadata


def save_run_artifacts(
    *,
    script_path: Path,
    loss_tracker: LossTracker,
    sample_text: str,
    metadata: dict[str, object],
    artifacts_root: Path | None = None,
) -> RunArtifacts:
    """Write the loss history, sample, and metadata into one artifact directory."""
    loss_history_csv, loss_curve_svg = loss_tracker.save(
        script_path=script_path,
        artifacts_root=artifacts_root,
    )
    run_dir = loss_history_csv.parent

    sample_path = run_dir / "sample.txt"
    sample_path.write_text(sample_text + "\n", encoding="utf-8")

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(serialize_mapping(metadata), indent=2) + "\n",
        encoding="utf-8",
    )

    return RunArtifacts(
        run_dir=run_dir,
        loss_history_csv=loss_history_csv,
        loss_curve_svg=loss_curve_svg,
        sample_path=sample_path,
        metadata_path=metadata_path,
    )


def print_run_summary(
    *,
    metadata: dict[str, object],
    artifacts: RunArtifacts,
    sample_text: str,
) -> None:
    """Print the minimal metrics needed to compare training runs quickly."""
    del artifacts
    del sample_text

    train_loss = metadata.get("final_train_loss")
    validation_loss = metadata.get("final_validation_loss")
    if validation_loss is None:
        validation_loss = metadata.get("final_validation_subset_loss")
    tokens_per_second = metadata.get("tokens_per_second")

    if train_loss is not None:
        if isinstance(train_loss, float):
            print(f"train_loss={train_loss:.6f}")
        else:
            print(f"train_loss={train_loss}")

    if validation_loss is not None:
        if isinstance(validation_loss, float):
            print(f"validation_loss={validation_loss:.6f}")
        else:
            print(f"validation_loss={validation_loss}")

    if tokens_per_second is not None:
        if isinstance(tokens_per_second, float):
            print(f"tokens_per_second={tokens_per_second:.3f}")
        else:
            print(f"tokens_per_second={tokens_per_second}")


def serialize_mapping(values: dict[str, object]) -> dict[str, object]:
    """Convert a dictionary into JSON-safe plain Python values."""
    return {key: serialize_for_json(value) for key, value in values.items()}


def serialize_for_json(value: object) -> Any:
    """Convert paths, dataclasses, and containers into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]
    return value
