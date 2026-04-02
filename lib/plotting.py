"""Save simple loss histories and SVG curves for experiment inspection."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pathlib import Path
from typing import Sequence

DEFAULT_ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "experiments"
SVG_HEIGHT = 400
SVG_WIDTH = 900


@dataclass
class LossTracker:
    """Collect train, optional train-subset, and validation-subset losses."""

    print_updates: bool = True
    train_steps: list[int] = field(default_factory=list)
    train_subset_steps: list[int] = field(default_factory=list)
    validation_subset_steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    train_subset_losses: list[float] = field(default_factory=list)
    validation_subset_losses: list[float] = field(default_factory=list)

    def log(
        self,
        *,
        step: int,
        train_loss: float,
        validation_subset_loss: float,
        train_subset_loss: float | None = None,
    ) -> None:
        """Record one training step and its matching evaluation estimates."""
        if step <= 0:
            raise ValueError("step must be positive")

        self.train_steps.append(step)
        self.validation_subset_steps.append(step)
        self.train_losses.append(train_loss)
        self.validation_subset_losses.append(validation_subset_loss)
        if train_subset_loss is not None:
            self.train_subset_steps.append(step)
            self.train_subset_losses.append(train_subset_loss)
        if self.print_updates:
            line = (
                f"step={step} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={validation_subset_loss:.6f}"
            )
            print(line)

    def save(
        self,
        *,
        script_path: Path,
        artifacts_root: Path | None = None,
    ) -> tuple[Path, Path]:
        """Write the tracked losses to CSV and SVG artifacts."""
        return save_loss_artifacts(
            script_path=script_path,
            artifacts_root=artifacts_root,
            train_steps=self.train_steps,
            train_subset_steps=self.train_subset_steps if self.train_subset_steps else None,
            validation_subset_steps=self.validation_subset_steps,
            train_losses=self.train_losses,
            train_subset_losses=self.train_subset_losses if self.train_subset_losses else None,
            validation_subset_losses=self.validation_subset_losses,
        )


def save_loss_artifacts(
    *,
    script_path: Path,
    artifacts_root: Path | None = None,
    train_steps: Sequence[int],
    train_subset_steps: Sequence[int] | None,
    validation_subset_steps: Sequence[int],
    train_losses: Sequence[float],
    train_subset_losses: Sequence[float] | None,
    validation_subset_losses: Sequence[float],
) -> tuple[Path, Path]:
    """Persist loss history in a timestamped artifact directory."""
    _validate_series("train", train_steps, train_losses)
    if train_subset_steps is not None and train_subset_losses is not None:
        _validate_series("train_subset", train_subset_steps, train_subset_losses)
    _validate_series("validation_subset", validation_subset_steps, validation_subset_losses)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = resolve_artifacts_root(artifacts_root) / script_path.stem / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    csv_path = run_dir / "loss_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "step", "loss"])
        writer.writerows(("train", step, loss) for step, loss in zip(train_steps, train_losses))
        if train_subset_steps is not None and train_subset_losses is not None:
            writer.writerows(
                ("train_subset", step, loss)
                for step, loss in zip(train_subset_steps, train_subset_losses)
            )
        writer.writerows(
            ("validation_subset", step, loss)
            for step, loss in zip(validation_subset_steps, validation_subset_losses)
        )

    svg_path = run_dir / "loss_curve.svg"
    svg_path.write_text(
        _build_loss_curve_svg(
            train_steps=train_steps,
            train_losses=train_losses,
            train_subset_steps=train_subset_steps,
            train_subset_losses=train_subset_losses,
            validation_subset_steps=validation_subset_steps,
            validation_subset_losses=validation_subset_losses,
        ),
        encoding="utf-8",
    )

    return csv_path, svg_path


def resolve_artifacts_root(artifacts_root: Path | None = None) -> Path:
    """Choose the artifact root from an override, env var, or repo default."""
    if artifacts_root is not None:
        return artifacts_root

    env_artifacts_root = os.environ.get("LLM_LAB_ARTIFACTS_ROOT")
    if env_artifacts_root:
        return Path(env_artifacts_root)

    return DEFAULT_ARTIFACTS_ROOT


def _validate_series(name: str, steps: Sequence[int], losses: Sequence[float]) -> None:
    """Check that one plotted loss series is non-empty and aligned."""
    if not steps:
        raise ValueError(f"{name} steps must contain at least one point")
    if len(steps) != len(losses):
        raise ValueError(f"{name} steps and losses must have the same length")


def _build_loss_curve_svg(
    *,
    train_steps: Sequence[int],
    train_losses: Sequence[float],
    train_subset_steps: Sequence[int] | None,
    train_subset_losses: Sequence[float] | None,
    validation_subset_steps: Sequence[int],
    validation_subset_losses: Sequence[float],
) -> str:
    """Render a minimal SVG loss chart for quick experiment review."""
    left_pad = 64
    right_pad = 24
    top_pad = 24
    bottom_pad = 44
    plot_width = SVG_WIDTH - left_pad - right_pad
    plot_height = SVG_HEIGHT - top_pad - bottom_pad

    all_steps = [*train_steps, *validation_subset_steps]
    all_losses = [*train_losses, *validation_subset_losses]
    if train_subset_steps is not None and train_subset_losses is not None:
        all_steps.extend(train_subset_steps)
        all_losses.extend(train_subset_losses)
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    if math.isclose(min_loss, max_loss):
        padding = max(abs(min_loss) * 0.1, 0.1)
        min_loss -= padding
        max_loss += padding

    step_min = min(all_steps)
    step_max = max(all_steps)
    step_span = max(step_max - step_min, 1)
    loss_span = max(max_loss - min_loss, 1e-6)

    def point(step: int, loss: float) -> tuple[float, float]:
        """Map one loss point into SVG coordinates."""
        x = left_pad + ((step - step_min) / step_span) * plot_width
        y = top_pad + ((max_loss - loss) / loss_span) * plot_height
        return x, y

    def polyline(steps: Sequence[int], losses: Sequence[float]) -> str:
        """Convert one loss series into SVG polyline coordinates."""
        return " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (point(step, loss) for step, loss in zip(steps, losses))
        )

    raw_train_points = polyline(train_steps, train_losses)
    train_subset_points = ""
    if train_subset_steps is not None and train_subset_losses is not None:
        train_subset_points = polyline(train_subset_steps, train_subset_losses)
    validation_subset_points = polyline(validation_subset_steps, validation_subset_losses)

    last_validation_subset_x, last_validation_subset_y = point(
        validation_subset_steps[-1],
        validation_subset_losses[-1],
    )
    last_train_x, last_train_y = point(train_steps[-1], train_losses[-1])
    train_subset_legend = ""
    train_subset_polyline = ""
    train_subset_circle = ""
    if train_subset_steps is not None and train_subset_losses is not None:
        last_train_subset_x, last_train_subset_y = point(
            train_subset_steps[-1],
            train_subset_losses[-1],
        )
        train_subset_polyline = (
            f'  <polyline points="{train_subset_points}" stroke="#059669" '
            f'stroke-width="2" fill="none"/>\n'
        )
        train_subset_circle = (
            f'  <circle cx="{last_train_subset_x:.2f}" cy="{last_train_subset_y:.2f}" '
            f'r="3" fill="#059669"/>\n'
        )
        train_subset_legend = (
            f'  <text x="{SVG_WIDTH - 220}" y="38" fill="#57606a" '
            f'font-family="monospace" font-size="12">train subset</text>\n'
            f'  <line x1="{SVG_WIDTH - 300}" y1="34" x2="{SVG_WIDTH - 228}" y2="34" '
            f'stroke="#059669" stroke-width="2"/>\n'
        )
        validation_legend_y = 58
        validation_line_y = 54
        title = "Train, train subset, and validation subset loss vs. step"
    else:
        validation_legend_y = 38
        validation_line_y = 34
        title = "Train and validation subset loss vs. step"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" fill="none">
  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white"/>
  <line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{SVG_HEIGHT - bottom_pad}" stroke="#d0d7de" stroke-width="1"/>
  <line x1="{left_pad}" y1="{SVG_HEIGHT - bottom_pad}" x2="{SVG_WIDTH - right_pad}" y2="{SVG_HEIGHT - bottom_pad}" stroke="#d0d7de" stroke-width="1"/>
  <text x="{left_pad}" y="16" fill="#24292f" font-family="monospace" font-size="14">{title}</text>
  <text x="{left_pad}" y="{SVG_HEIGHT - 12}" fill="#57606a" font-family="monospace" font-size="12">step {step_min} to {step_max}</text>
  <text x="12" y="{top_pad + 4}" fill="#57606a" font-family="monospace" font-size="12">{max_loss:.4f}</text>
  <text x="12" y="{SVG_HEIGHT - bottom_pad + 4}" fill="#57606a" font-family="monospace" font-size="12">{min_loss:.4f}</text>
  <polyline points="{raw_train_points}" stroke="#2563eb" stroke-width="2" fill="none"/>
{train_subset_polyline}  <polyline points="{validation_subset_points}" stroke="#dc2626" stroke-width="2" fill="none"/>
  <circle cx="{last_train_x:.2f}" cy="{last_train_y:.2f}" r="3" fill="#2563eb"/>
{train_subset_circle}  <circle cx="{last_validation_subset_x:.2f}" cy="{last_validation_subset_y:.2f}" r="3" fill="#dc2626"/>
  <text x="{SVG_WIDTH - 220}" y="18" fill="#57606a" font-family="monospace" font-size="12">train</text>
  <line x1="{SVG_WIDTH - 300}" y1="14" x2="{SVG_WIDTH - 228}" y2="14" stroke="#2563eb" stroke-width="2"/>
{train_subset_legend}  <text x="{SVG_WIDTH - 220}" y="{validation_legend_y}" fill="#57606a" font-family="monospace" font-size="12">validation subset</text>
  <line x1="{SVG_WIDTH - 300}" y1="{validation_line_y}" x2="{SVG_WIDTH - 228}" y2="{validation_line_y}" stroke="#dc2626" stroke-width="2"/>
</svg>
"""
