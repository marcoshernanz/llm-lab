from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
from typing import Sequence

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "experiments"
SVG_HEIGHT = 400
SVG_WIDTH = 900


@dataclass
class LossTracker:
    log_interval: int
    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)

    def log(self, *, step: int, train_loss: float, validation_loss: float) -> None:
        if step <= 0:
            raise ValueError("step must be positive")

        self.train_losses.append(train_loss)
        self.validation_losses.append(validation_loss)
        print(f"step={step} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}")

    def save(self, *, script_path: Path) -> tuple[Path, Path]:
        return save_loss_artifacts(
            script_path=script_path,
            train_losses=self.train_losses,
            validation_losses=self.validation_losses,
            train_log_interval=self.log_interval,
            validation_log_interval=self.log_interval,
        )


def save_loss_artifacts(
    *,
    script_path: Path,
    train_losses: Sequence[float],
    validation_losses: Sequence[float],
    train_log_interval: int,
    validation_log_interval: int,
) -> tuple[Path, Path]:
    if train_log_interval <= 0:
        raise ValueError("train_log_interval must be positive")
    if validation_log_interval <= 0:
        raise ValueError("validation_log_interval must be positive")
    if not train_losses:
        raise ValueError("train_losses must contain at least one point")
    if not validation_losses:
        raise ValueError("validation_losses must contain at least one point")

    train_steps = _build_steps(len(train_losses), train_log_interval)
    validation_steps = _build_steps(len(validation_losses), validation_log_interval)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ARTIFACTS_ROOT / script_path.stem / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    csv_path = run_dir / "loss_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "step", "loss"])
        writer.writerows(("train", step, loss) for step, loss in zip(train_steps, train_losses))
        writer.writerows(
            ("validation", step, loss) for step, loss in zip(validation_steps, validation_losses)
        )

    svg_path = run_dir / "loss_curve.svg"
    svg_path.write_text(
        _build_loss_curve_svg(
            train_steps=train_steps,
            train_losses=train_losses,
            validation_steps=validation_steps,
            validation_losses=validation_losses,
        ),
        encoding="utf-8",
    )

    return csv_path, svg_path


def _build_steps(num_points: int, interval: int) -> list[int]:
    return [interval * (index + 1) for index in range(num_points)]


def _build_loss_curve_svg(
    *,
    train_steps: Sequence[int],
    train_losses: Sequence[float],
    validation_steps: Sequence[int],
    validation_losses: Sequence[float],
) -> str:
    left_pad = 64
    right_pad = 24
    top_pad = 24
    bottom_pad = 44
    plot_width = SVG_WIDTH - left_pad - right_pad
    plot_height = SVG_HEIGHT - top_pad - bottom_pad

    all_steps = [*train_steps, *validation_steps]
    all_losses = [*train_losses, *validation_losses]
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
        x = left_pad + ((step - step_min) / step_span) * plot_width
        y = top_pad + ((max_loss - loss) / loss_span) * plot_height
        return x, y

    def polyline(steps: Sequence[int], losses: Sequence[float]) -> str:
        return " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (point(step, loss) for step, loss in zip(steps, losses))
        )

    smoothed_train_losses = _ema(train_losses, decay=0.9)
    raw_train_points = polyline(train_steps, train_losses)
    smooth_train_points = polyline(train_steps, smoothed_train_losses)
    validation_points = polyline(validation_steps, validation_losses)

    last_validation_x, last_validation_y = point(validation_steps[-1], validation_losses[-1])
    last_train_x, last_train_y = point(train_steps[-1], smoothed_train_losses[-1])

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" fill="none">
  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white"/>
  <line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{SVG_HEIGHT - bottom_pad}" stroke="#d0d7de" stroke-width="1"/>
  <line x1="{left_pad}" y1="{SVG_HEIGHT - bottom_pad}" x2="{SVG_WIDTH - right_pad}" y2="{SVG_HEIGHT - bottom_pad}" stroke="#d0d7de" stroke-width="1"/>
  <text x="{left_pad}" y="16" fill="#24292f" font-family="monospace" font-size="14">Train and validation loss vs. step</text>
  <text x="{left_pad}" y="{SVG_HEIGHT - 12}" fill="#57606a" font-family="monospace" font-size="12">step {step_min} to {step_max}</text>
  <text x="12" y="{top_pad + 4}" fill="#57606a" font-family="monospace" font-size="12">{max_loss:.4f}</text>
  <text x="12" y="{SVG_HEIGHT - bottom_pad + 4}" fill="#57606a" font-family="monospace" font-size="12">{min_loss:.4f}</text>
  <polyline points="{raw_train_points}" stroke="#cbd5e1" stroke-width="1" fill="none"/>
  <polyline points="{smooth_train_points}" stroke="#2563eb" stroke-width="2" fill="none"/>
  <polyline points="{validation_points}" stroke="#dc2626" stroke-width="2" fill="none"/>
  <circle cx="{last_train_x:.2f}" cy="{last_train_y:.2f}" r="3" fill="#2563eb"/>
  <circle cx="{last_validation_x:.2f}" cy="{last_validation_y:.2f}" r="3" fill="#dc2626"/>
  <text x="{SVG_WIDTH - 220}" y="18" fill="#57606a" font-family="monospace" font-size="12">raw train</text>
  <line x1="{SVG_WIDTH - 300}" y1="14" x2="{SVG_WIDTH - 228}" y2="14" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="{SVG_WIDTH - 220}" y="38" fill="#57606a" font-family="monospace" font-size="12">smoothed train</text>
  <line x1="{SVG_WIDTH - 300}" y1="34" x2="{SVG_WIDTH - 228}" y2="34" stroke="#2563eb" stroke-width="2"/>
  <text x="{SVG_WIDTH - 220}" y="58" fill="#57606a" font-family="monospace" font-size="12">validation</text>
  <line x1="{SVG_WIDTH - 300}" y1="54" x2="{SVG_WIDTH - 228}" y2="54" stroke="#dc2626" stroke-width="2"/>
</svg>
"""


def _ema(values: Sequence[float], decay: float) -> list[float]:
    smoothed_values: list[float] = []
    ema: float | None = None

    for value in values:
        ema = value if ema is None else decay * ema + (1.0 - decay) * value
        smoothed_values.append(ema)

    return smoothed_values
