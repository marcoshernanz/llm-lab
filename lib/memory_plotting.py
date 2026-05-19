"""Save memory-experiment metrics as CSV files and simple SVG curves."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pathlib import Path
from typing import Sequence

DEFAULT_ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "memory_experiments"
SVG_HEIGHT = 400
SVG_WIDTH = 900


@dataclass(frozen=True)
class MemoryMetricRecord:
    """One evaluation checkpoint from a memory experiment."""

    step: int
    batch_answer_loss: float
    eval_answer_loss: float
    eval_exact_answer_accuracy: float
    eval_candidate_value_accuracy: float
    eval_mean_address_movement: float | None = None
    eval_mean_memory_usage: float | None = None
    eval_usage_entropy: float | None = None
    eval_mean_allocation_gate: float | None = None
    eval_mean_write_gate: float | None = None


@dataclass(frozen=True)
class MetricSeries:
    """One named curve in an SVG chart."""

    name: str
    values: Sequence[float]
    color: str


@dataclass(frozen=True)
class Baseline:
    """One horizontal reference line in an SVG chart."""

    name: str
    value: float
    color: str


@dataclass
class MemoryMetricTracker:
    """Collect and persist the metrics used by memory-architecture experiments."""

    print_updates: bool = True
    candidate_guess_exact_baseline: float | None = None
    random_value_exact_baseline: float | None = None
    random_value_candidate_baseline: float | None = None
    records: list[MemoryMetricRecord] = field(default_factory=list)

    def log(
        self,
        *,
        step: int,
        batch_answer_loss: float,
        eval_answer_loss: float,
        eval_exact_answer_accuracy: float,
        eval_candidate_value_accuracy: float,
        eval_mean_address_movement: float | None = None,
        eval_mean_memory_usage: float | None = None,
        eval_usage_entropy: float | None = None,
        eval_mean_allocation_gate: float | None = None,
        eval_mean_write_gate: float | None = None,
    ) -> None:
        """Record one memory-experiment checkpoint and optionally print it."""
        record = MemoryMetricRecord(
            step=step,
            batch_answer_loss=_checked_float("batch_answer_loss", batch_answer_loss),
            eval_answer_loss=_checked_float("eval_answer_loss", eval_answer_loss),
            eval_exact_answer_accuracy=_checked_float(
                "eval_exact_answer_accuracy",
                eval_exact_answer_accuracy,
            ),
            eval_candidate_value_accuracy=_checked_float(
                "eval_candidate_value_accuracy",
                eval_candidate_value_accuracy,
            ),
            eval_mean_address_movement=(
                None
                if eval_mean_address_movement is None
                else _checked_float("eval_mean_address_movement", eval_mean_address_movement)
            ),
            eval_mean_memory_usage=(
                None
                if eval_mean_memory_usage is None
                else _checked_float("eval_mean_memory_usage", eval_mean_memory_usage)
            ),
            eval_usage_entropy=(
                None
                if eval_usage_entropy is None
                else _checked_float("eval_usage_entropy", eval_usage_entropy)
            ),
            eval_mean_allocation_gate=(
                None
                if eval_mean_allocation_gate is None
                else _checked_float("eval_mean_allocation_gate", eval_mean_allocation_gate)
            ),
            eval_mean_write_gate=(
                None
                if eval_mean_write_gate is None
                else _checked_float("eval_mean_write_gate", eval_mean_write_gate)
            ),
        )
        if record.step <= 0:
            raise ValueError("step must be positive")

        self.records.append(record)
        if self.print_updates:
            print(_format_record(record))

    def save(
        self,
        *,
        script_path: Path,
        artifacts_root: Path | None = None,
    ) -> dict[str, Path]:
        """Write metrics to CSV and SVG artifacts."""
        return save_memory_metric_artifacts(
            script_path=script_path,
            artifacts_root=artifacts_root,
            records=self.records,
            candidate_guess_exact_baseline=self.candidate_guess_exact_baseline,
            random_value_exact_baseline=self.random_value_exact_baseline,
            random_value_candidate_baseline=self.random_value_candidate_baseline,
        )


def save_memory_metric_artifacts(
    *,
    script_path: Path,
    artifacts_root: Path | None = None,
    records: Sequence[MemoryMetricRecord],
    candidate_guess_exact_baseline: float | None = None,
    random_value_exact_baseline: float | None = None,
    random_value_candidate_baseline: float | None = None,
) -> dict[str, Path]:
    """Persist memory metrics in a timestamped artifact directory."""
    _validate_records(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = resolve_memory_artifacts_root(artifacts_root) / script_path.stem / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    artifacts: dict[str, Path] = {}
    csv_path = run_dir / "metrics.csv"
    _write_metrics_csv(csv_path, records)
    artifacts["metrics_csv"] = csv_path

    steps = [record.step for record in records]
    loss_path = run_dir / "loss_curve.svg"
    loss_path.write_text(
        _build_metric_curve_svg(
            title="Memory experiment loss vs. step",
            steps=steps,
            series=[
                MetricSeries(
                    "batch answer loss",
                    [record.batch_answer_loss for record in records],
                    "#2563eb",
                ),
                MetricSeries(
                    "eval answer loss",
                    [record.eval_answer_loss for record in records],
                    "#dc2626",
                ),
            ],
        ),
        encoding="utf-8",
    )
    artifacts["loss_curve_svg"] = loss_path

    accuracy_baselines = _accuracy_baselines(
        candidate_guess_exact_baseline=candidate_guess_exact_baseline,
        random_value_exact_baseline=random_value_exact_baseline,
        random_value_candidate_baseline=random_value_candidate_baseline,
    )
    accuracy_path = run_dir / "accuracy_curve.svg"
    accuracy_path.write_text(
        _build_metric_curve_svg(
            title="Memory experiment accuracy vs. step",
            steps=steps,
            series=[
                MetricSeries(
                    "exact answer",
                    [record.eval_exact_answer_accuracy for record in records],
                    "#059669",
                ),
                MetricSeries(
                    "candidate value",
                    [record.eval_candidate_value_accuracy for record in records],
                    "#7c3aed",
                ),
            ],
            baselines=accuracy_baselines,
            y_min=0.0,
            y_max=1.0,
        ),
        encoding="utf-8",
    )
    artifacts["accuracy_curve_svg"] = accuracy_path

    if any(record.eval_mean_address_movement is not None for record in records):
        movement_path = run_dir / "address_movement_curve.svg"
        movement_path.write_text(
            _build_metric_curve_svg(
                title="Mean address movement vs. step",
                steps=steps,
                series=[
                    MetricSeries(
                        "mean address movement",
                        [
                            record.eval_mean_address_movement
                            if record.eval_mean_address_movement is not None
                            else 0.0
                            for record in records
                        ],
                        "#ea580c",
                    )
                ],
                y_min=0.0,
            ),
            encoding="utf-8",
        )
        artifacts["address_movement_curve_svg"] = movement_path

    if any(record.eval_mean_memory_usage is not None for record in records):
        usage_path = run_dir / "usage_curve.svg"
        usage_path.write_text(
            _build_metric_curve_svg(
                title="Memory usage diagnostics vs. step",
                steps=steps,
                series=[
                    MetricSeries(
                        "mean usage",
                        [
                            record.eval_mean_memory_usage
                            if record.eval_mean_memory_usage is not None
                            else 0.0
                            for record in records
                        ],
                        "#0891b2",
                    ),
                    MetricSeries(
                        "usage entropy",
                        [
                            record.eval_usage_entropy
                            if record.eval_usage_entropy is not None
                            else 0.0
                            for record in records
                        ],
                        "#9333ea",
                    ),
                    MetricSeries(
                        "allocation gate",
                        [
                            record.eval_mean_allocation_gate
                            if record.eval_mean_allocation_gate is not None
                            else 0.0
                            for record in records
                        ],
                        "#ea580c",
                    ),
                    MetricSeries(
                        "write gate",
                        [
                            record.eval_mean_write_gate
                            if record.eval_mean_write_gate is not None
                            else 0.0
                            for record in records
                        ],
                        "#16a34a",
                    ),
                ],
                y_min=0.0,
                y_max=1.0,
            ),
            encoding="utf-8",
        )
        artifacts["usage_curve_svg"] = usage_path

    return artifacts


def resolve_memory_artifacts_root(artifacts_root: Path | None = None) -> Path:
    """Choose the artifact root from an override, env var, or repo default."""
    if artifacts_root is not None:
        return artifacts_root

    env_artifacts_root = os.environ.get("LLM_LAB_MEMORY_ARTIFACTS_ROOT")
    if env_artifacts_root:
        return Path(env_artifacts_root)

    return DEFAULT_ARTIFACTS_ROOT


def _write_metrics_csv(path: Path, records: Sequence[MemoryMetricRecord]) -> None:
    """Write one row per evaluation checkpoint."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "step",
                "batch_answer_loss",
                "eval_answer_loss",
                "eval_exact_answer_accuracy",
                "eval_candidate_value_accuracy",
                "eval_mean_address_movement",
                "eval_mean_memory_usage",
                "eval_usage_entropy",
                "eval_mean_allocation_gate",
                "eval_mean_write_gate",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.step,
                    record.batch_answer_loss,
                    record.eval_answer_loss,
                    record.eval_exact_answer_accuracy,
                    record.eval_candidate_value_accuracy,
                    "" if record.eval_mean_address_movement is None else record.eval_mean_address_movement,
                    "" if record.eval_mean_memory_usage is None else record.eval_mean_memory_usage,
                    "" if record.eval_usage_entropy is None else record.eval_usage_entropy,
                    "" if record.eval_mean_allocation_gate is None else record.eval_mean_allocation_gate,
                    "" if record.eval_mean_write_gate is None else record.eval_mean_write_gate,
                ]
            )


def _build_metric_curve_svg(
    *,
    title: str,
    steps: Sequence[int],
    series: Sequence[MetricSeries],
    baselines: Sequence[Baseline] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    """Render one multi-series SVG chart."""
    _validate_chart_inputs(steps, series)
    baselines = baselines or []

    left_pad = 64
    right_pad = 24
    top_pad = 24
    bottom_pad = 44
    plot_width = SVG_WIDTH - left_pad - right_pad
    plot_height = SVG_HEIGHT - top_pad - bottom_pad

    all_values = [value for metric in series for value in metric.values]
    all_values.extend(baseline.value for baseline in baselines)
    min_value = min(all_values) if y_min is None else y_min
    max_value = max(all_values) if y_max is None else y_max
    if math.isclose(min_value, max_value):
        padding = max(abs(min_value) * 0.1, 0.1)
        min_value -= padding
        max_value += padding

    step_min = min(steps)
    step_max = max(steps)
    step_span = max(step_max - step_min, 1)
    value_span = max(max_value - min_value, 1e-6)

    def point(step: int, value: float) -> tuple[float, float]:
        """Map one metric point into SVG coordinates."""
        x = left_pad + ((step - step_min) / step_span) * plot_width
        y = top_pad + ((max_value - value) / value_span) * plot_height
        return x, y

    def polyline(values: Sequence[float]) -> str:
        """Convert one series into SVG polyline coordinates."""
        return " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (point(step, value) for step, value in zip(steps, values))
        )

    chart_lines = [
        f'<polyline points="{polyline(metric.values)}" stroke="{metric.color}" '
        'stroke-width="2" fill="none"/>'
        for metric in series
    ]
    chart_lines.extend(_baseline_svg(baseline, point, step_min, step_max) for baseline in baselines)

    endpoint_circles = []
    for metric in series:
        last_x, last_y = point(steps[-1], metric.values[-1])
        endpoint_circles.append(
            f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="3" fill="{metric.color}"/>'
        )

    legend_lines = _legend_svg(series, baselines)
    chart_body = "\n  ".join([*chart_lines, *endpoint_circles, *legend_lines])

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" fill="none">
  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white"/>
  <line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{SVG_HEIGHT - bottom_pad}" stroke="#d0d7de" stroke-width="1"/>
  <line x1="{left_pad}" y1="{SVG_HEIGHT - bottom_pad}" x2="{SVG_WIDTH - right_pad}" y2="{SVG_HEIGHT - bottom_pad}" stroke="#d0d7de" stroke-width="1"/>
  <text x="{left_pad}" y="16" fill="#24292f" font-family="monospace" font-size="14">{_svg_escape(title)}</text>
  <text x="{left_pad}" y="{SVG_HEIGHT - 12}" fill="#57606a" font-family="monospace" font-size="12">step {step_min} to {step_max}</text>
  <text x="12" y="{top_pad + 4}" fill="#57606a" font-family="monospace" font-size="12">{max_value:.4f}</text>
  <text x="12" y="{SVG_HEIGHT - bottom_pad + 4}" fill="#57606a" font-family="monospace" font-size="12">{min_value:.4f}</text>
  {chart_body}
</svg>
"""


def _baseline_svg(
    baseline: Baseline,
    point,
    step_min: int,
    step_max: int,
) -> str:
    """Render one horizontal baseline line."""
    x1, y1 = point(step_min, baseline.value)
    x2, y2 = point(step_max, baseline.value)
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{baseline.color}" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )


def _legend_svg(series: Sequence[MetricSeries], baselines: Sequence[Baseline]) -> list[str]:
    """Build a compact legend in the top-right of the SVG."""
    legend_items = [(metric.name, metric.color, False) for metric in series]
    legend_items.extend((baseline.name, baseline.color, True) for baseline in baselines)

    lines = []
    legend_x = SVG_WIDTH - 315
    text_x = SVG_WIDTH - 225
    for index, (name, color, dashed) in enumerate(legend_items):
        y = 18 + index * 20
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        lines.append(
            f'<line x1="{legend_x}" y1="{y - 4}" x2="{text_x - 8}" y2="{y - 4}" '
            f'stroke="{color}" stroke-width="2"{dash}/>'
        )
        lines.append(
            f'<text x="{text_x}" y="{y}" fill="#57606a" font-family="monospace" '
            f'font-size="12">{_svg_escape(name)}</text>'
        )
    return lines


def _accuracy_baselines(
    *,
    candidate_guess_exact_baseline: float | None,
    random_value_exact_baseline: float | None,
    random_value_candidate_baseline: float | None,
) -> list[Baseline]:
    """Create horizontal reference lines for the memory accuracy chart."""
    baselines = []
    if candidate_guess_exact_baseline is not None:
        baselines.append(
            Baseline("candidate guess exact", candidate_guess_exact_baseline, "#9ca3af")
        )
    if random_value_exact_baseline is not None:
        baselines.append(Baseline("random value exact", random_value_exact_baseline, "#64748b"))
    if random_value_candidate_baseline is not None:
        baselines.append(
            Baseline("random value candidate", random_value_candidate_baseline, "#94a3b8")
        )
    return baselines


def _format_record(record: MemoryMetricRecord) -> str:
    """Format one checkpoint to match the existing memory experiment logs."""
    line = (
        f"step={record.step} "
        f"batch_answer_loss={record.batch_answer_loss:.4f} "
        f"eval_answer_loss={record.eval_answer_loss:.4f} "
        f"eval_exact_answer_accuracy={record.eval_exact_answer_accuracy:.4f} "
        f"eval_candidate_value_accuracy={record.eval_candidate_value_accuracy:.4f}"
    )
    if record.eval_mean_address_movement is not None:
        line += f" eval_mean_address_movement={record.eval_mean_address_movement:.6f}"
    if record.eval_mean_memory_usage is not None:
        line += f" eval_mean_memory_usage={record.eval_mean_memory_usage:.4f}"
    if record.eval_usage_entropy is not None:
        line += f" eval_usage_entropy={record.eval_usage_entropy:.4f}"
    if record.eval_mean_allocation_gate is not None:
        line += f" eval_mean_allocation_gate={record.eval_mean_allocation_gate:.4f}"
    if record.eval_mean_write_gate is not None:
        line += f" eval_mean_write_gate={record.eval_mean_write_gate:.4f}"
    return line


def _validate_records(records: Sequence[MemoryMetricRecord]) -> None:
    """Check that a run has at least one well-formed checkpoint."""
    if not records:
        raise ValueError("records must contain at least one checkpoint")
    for record in records:
        if record.step <= 0:
            raise ValueError("all steps must be positive")


def _validate_chart_inputs(steps: Sequence[int], series: Sequence[MetricSeries]) -> None:
    """Check that all series are aligned with the step axis."""
    if not steps:
        raise ValueError("steps must contain at least one point")
    if not series:
        raise ValueError("series must contain at least one curve")
    for metric in series:
        if len(metric.values) != len(steps):
            raise ValueError(f"{metric.name} values must match step count")


def _checked_float(name: str, value: float) -> float:
    """Convert a metric to float and reject NaN or infinity."""
    checked_value = float(value)
    if not math.isfinite(checked_value):
        raise ValueError(f"{name} must be finite")
    return checked_value


def _svg_escape(value: str) -> str:
    """Escape the small amount of text we place in generated SVGs."""
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
