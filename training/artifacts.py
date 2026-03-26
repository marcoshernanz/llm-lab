from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any


SVG_HEIGHT = 360
SVG_WIDTH = 800


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_dir: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    loss_curve_path: Path
    samples_dir: Path


@dataclass(frozen=True, slots=True)
class MetricRow:
    step: int
    raw_train_loss: float
    ema_train_loss: float
    validation_loss: float | None


def create_run_paths(output_root: Path, experiment_name: str) -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / experiment_name / timestamp
    return ensure_run_paths(run_dir, create=True)


def ensure_run_paths(run_dir: Path, *, create: bool) -> RunPaths:
    if create:
        run_dir.mkdir(parents=True, exist_ok=False)
    samples_dir = run_dir / "samples"
    if create:
        samples_dir.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.toml",
        metadata_path=run_dir / "run_metadata.json",
        metrics_path=run_dir / "metrics.csv",
        loss_curve_path=run_dir / "loss_curve.svg",
        samples_dir=samples_dir,
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_metric_row(path: Path, row: MetricRow) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(["step", "raw_train_loss", "ema_train_loss", "validation_loss"])
        writer.writerow(
            [
                row.step,
                f"{row.raw_train_loss:.12f}",
                f"{row.ema_train_loss:.12f}",
                "" if row.validation_loss is None else f"{row.validation_loss:.12f}",
            ]
        )


def read_metric_rows(path: Path) -> list[MetricRow]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[MetricRow] = []
        for row in reader:
            validation_loss = row["validation_loss"]
            rows.append(
                MetricRow(
                    step=int(row["step"]),
                    raw_train_loss=float(row["raw_train_loss"]),
                    ema_train_loss=float(row["ema_train_loss"]),
                    validation_loss=None if validation_loss == "" else float(validation_loss),
                )
            )
    return rows


def write_sample(samples_dir: Path, step: int, sample_text: str) -> Path:
    sample_path = samples_dir / f"step_{step:06d}.txt"
    sample_path.write_text(sample_text, encoding="utf-8")
    return sample_path


def regenerate_loss_curve(metrics_path: Path, svg_path: Path) -> None:
    loss_history = read_metric_rows(metrics_path)
    if not loss_history:
        return
    svg_path.write_text(_build_loss_curve_svg(loss_history), encoding="utf-8")


def _build_loss_curve_svg(loss_history: list[MetricRow]) -> str:
    left_pad = 64
    right_pad = 24
    top_pad = 24
    bottom_pad = 44
    plot_width = SVG_WIDTH - left_pad - right_pad
    plot_height = SVG_HEIGHT - top_pad - bottom_pad

    steps = [row.step for row in loss_history]
    raw_losses = [row.raw_train_loss for row in loss_history]
    smooth_losses = [row.ema_train_loss for row in loss_history]

    min_loss = min(min(raw_losses), min(smooth_losses))
    max_loss = max(max(raw_losses), max(smooth_losses))
    if math.isclose(min_loss, max_loss):
        padding = max(abs(min_loss) * 0.1, 0.1)
        min_loss -= padding
        max_loss += padding

    step_min = steps[0]
    step_max = steps[-1]
    step_span = max(step_max - step_min, 1)
    loss_span = max(max_loss - min_loss, 1e-6)

    def point(step: int, loss: float) -> tuple[float, float]:
        x = left_pad + ((step - step_min) / step_span) * plot_width
        y = top_pad + ((max_loss - loss) / loss_span) * plot_height
        return x, y

    def polyline(losses: list[float]) -> str:
        return " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (point(step, loss) for step, loss in zip(steps, losses))
        )

    raw_points = polyline(raw_losses)
    smooth_points = polyline(smooth_losses)
    first_x, first_raw_y = point(steps[0], raw_losses[0])
    last_x, last_smooth_y = point(steps[-1], smooth_losses[-1])

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" fill="none">
  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white"/>
  <line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{SVG_HEIGHT - bottom_pad}" stroke="#d0d7de" stroke-width="1"/>
  <line x1="{left_pad}" y1="{SVG_HEIGHT - bottom_pad}" x2="{SVG_WIDTH - right_pad}" y2="{SVG_HEIGHT - bottom_pad}" stroke="#d0d7de" stroke-width="1"/>
  <text x="{left_pad}" y="16" fill="#24292f" font-family="monospace" font-size="14">Loss vs. step</text>
  <text x="{left_pad}" y="{SVG_HEIGHT - 12}" fill="#57606a" font-family="monospace" font-size="12">step {step_min} to {step_max}</text>
  <text x="12" y="{top_pad + 4}" fill="#57606a" font-family="monospace" font-size="12">{max_loss:.4f}</text>
  <text x="12" y="{SVG_HEIGHT - bottom_pad + 4}" fill="#57606a" font-family="monospace" font-size="12">{min_loss:.4f}</text>
  <polyline points="{raw_points}" stroke="#94a3b8" stroke-width="1.5" fill="none"/>
  <polyline points="{smooth_points}" stroke="#2563eb" stroke-width="2" fill="none"/>
  <circle cx="{first_x:.2f}" cy="{first_raw_y:.2f}" r="3" fill="#94a3b8"/>
  <circle cx="{last_x:.2f}" cy="{last_smooth_y:.2f}" r="3" fill="#2563eb"/>
  <text x="{SVG_WIDTH - 188}" y="18" fill="#57606a" font-family="monospace" font-size="12">raw loss</text>
  <line x1="{SVG_WIDTH - 248}" y1="14" x2="{SVG_WIDTH - 196}" y2="14" stroke="#94a3b8" stroke-width="1.5"/>
  <text x="{SVG_WIDTH - 188}" y="38" fill="#57606a" font-family="monospace" font-size="12">smooth loss</text>
  <line x1="{SVG_WIDTH - 248}" y1="34" x2="{SVG_WIDTH - 196}" y2="34" stroke="#2563eb" stroke-width="2"/>
</svg>
"""
