"""Save self-describing artifacts for reusable experiment runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
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
    config: object,
    execution_target: str | None,
    run_details: dict[str, object] | None = None,
    run_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assemble metadata so one artifact directory can explain one run."""
    config_values = serialize_for_json(config)
    if not isinstance(config_values, dict):
        raise TypeError("config must serialize to a dictionary of values")

    metadata: dict[str, object] = {
        "script_name": script_path.name,
        "execution_target": resolve_execution_target(execution_target),
        "jax_backend": jax.default_backend(),
        "jax_device_count": jax.device_count(),
        **config_values,
    }

    if run_details is not None:
        metadata.update(serialize_mapping(run_details))
    if run_metrics is not None:
        metadata.update(serialize_mapping(run_metrics))

    train_seconds = metadata.get("train_seconds")
    train_steps = metadata.get("train_steps")
    batch_size = metadata.get("batch_size")
    context_length = metadata.get("context_length")
    if isinstance(train_seconds, (int, float)) and train_seconds > 0:
        if isinstance(train_steps, (int, float)):
            steps_per_second = train_steps / train_seconds
            metadata["steps_per_second"] = steps_per_second
            if isinstance(batch_size, (int, float)) and isinstance(context_length, (int, float)):
                metadata["tokens_per_second"] = steps_per_second * batch_size * context_length

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
    """Print the most useful run fields in a stable, greppable format."""
    summary_keys = [
        "execution_target",
        "jax_backend",
        "jax_device_count",
        "token_shard_root",
        "tokenizer_path",
        "train_shards_used",
        "max_train_shards",
        "validation_shard_index",
        "train_subset_shard_index",
        "shard_mmap",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "train_steps",
        "train_chunk_length",
        "context_length",
        "embedding_dim",
        "hidden_dim",
        "num_heads",
        "num_decoder_blocks",
        "validation_subset_examples",
        "sample_tokens",
        "loaded_train_tokens",
        "loaded_train_subset_tokens",
        "loaded_validation_tokens",
        "final_train_loss",
        "final_train_subset_loss",
        "final_validation_subset_loss",
        "train_seconds",
        "steps_per_second",
        "tokens_per_second",
        "total_seconds",
    ]

    for key in summary_keys:
        if key not in metadata or metadata[key] is None:
            continue
        value = metadata[key]
        if key.startswith("final_") or key.endswith("_seconds") or key.endswith("_per_second"):
            if isinstance(value, float):
                value = f"{value:.6f}" if key.startswith("final_") else f"{value:.3f}"
        print(f"{key}={value}")

    print(f"loss_history_csv={artifacts['loss_history_csv']}")
    print(f"loss_curve_svg={artifacts['loss_curve_svg']}")
    print(f"sample_path={artifacts['sample_path']}")
    print(f"metadata_path={artifacts['metadata_path']}")
    print(f'sample="""\n{sample_text}\n"""')


def serialize_mapping(values: dict[str, object]) -> dict[str, object]:
    """Convert a dictionary into JSON-safe plain Python values."""
    return {key: serialize_for_json(value) for key, value in values.items()}


def serialize_for_json(value: object) -> Any:
    """Convert paths, dataclasses, and containers into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]
    return value
