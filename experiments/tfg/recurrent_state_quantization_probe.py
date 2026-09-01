"""Numerical stress test for repeatedly quantized Gated DeltaNet-style state.

This is deliberately not a model-quality or kernel benchmark.  It isolates the error-feedback path
created when a recurrent matrix is quantized after every write, and reports trajectory and final-state
error for uniform INT4 and INT8 storage.  The recurrence follows the scalar-decay Gated DeltaNet form;
KDA's per-key-channel decay is a richer case that this small diagnostic does not claim to cover.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    heads: int = 2
    key_dim: int = 32
    value_dim: int = 32
    sequence_lengths: tuple[int, ...] = (128, 512, 2048)
    decays: tuple[float, ...] = (0.99, 0.999)
    bits: tuple[int, ...] = (4, 8)
    seeds: tuple[int, ...] = (0, 1, 2)


def quantize_symmetric_per_head(state: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize and dequantize one state with a separate symmetric scale per head."""
    qmax = 2 ** (bits - 1) - 1
    scales = state.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12) / qmax
    return torch.round(state / scales).clamp(-qmax, qmax) * scales


def recurrent_trial(
    *, seed: int, steps: int, decay: float, bits: int, config: Config
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(seed)
    reference = torch.zeros(config.heads, config.key_dim, config.value_dim)
    quantized = reference.clone()
    output_error_squared = 0.0
    output_reference_squared = 0.0

    for _ in range(steps):
        key = torch.randn(config.heads, config.key_dim, generator=generator)
        key = torch.nn.functional.normalize(key, dim=-1)
        query = torch.randn(config.heads, config.key_dim, generator=generator)
        query = torch.nn.functional.normalize(query, dim=-1)
        value = torch.randn(config.heads, config.value_dim, generator=generator)
        beta = torch.sigmoid(torch.randn(config.heads, 1, generator=generator))

        def update(state: torch.Tensor) -> torch.Tensor:
            decayed = state * decay
            remembered = torch.einsum("hkv,hk->hv", decayed, key)
            delta = (value - remembered) * beta
            return decayed + torch.einsum("hk,hv->hkv", key, delta)

        reference = update(reference)
        quantized = quantize_symmetric_per_head(update(quantized), bits)
        reference_output = torch.einsum("hkv,hk->hv", reference, query)
        quantized_output = torch.einsum("hkv,hk->hv", quantized, query)
        output_error_squared += float((quantized_output - reference_output).square().sum())
        output_reference_squared += float(reference_output.square().sum())

    state_relative_l2 = float(
        torch.linalg.vector_norm(quantized - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    )
    trajectory_relative_l2 = float(
        np.sqrt(output_error_squared / max(output_reference_squared, 1e-24))
    )
    return {
        "seed": seed,
        "steps": steps,
        "decay": decay,
        "bits": bits,
        "state_relative_l2": state_relative_l2,
        "trajectory_relative_l2": trajectory_relative_l2,
    }


def storage_summary(config: Config, bits: int) -> dict[str, float | int]:
    elements = config.heads * config.key_dim * config.value_dim
    fp32_bytes = elements * 4
    payload_bytes = (elements * bits + 7) // 8
    scale_bytes = config.heads * 2  # one FP16 scale per head
    stored_bytes = payload_bytes + scale_bytes
    return {
        "bits": bits,
        "fp32_bytes": fp32_bytes,
        "quantized_payload_bytes": payload_bytes,
        "fp16_scale_bytes": scale_bytes,
        "stored_bytes": stored_bytes,
        "compression_ratio": fp32_bytes / stored_bytes,
    }


def aggregate(results: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    aggregates: list[dict[str, float | int]] = []
    keys = sorted({(int(row["steps"]), float(row["decay"]), int(row["bits"])) for row in results})
    for steps, decay, bits in keys:
        group = [
            row
            for row in results
            if row["steps"] == steps and row["decay"] == decay and row["bits"] == bits
        ]
        aggregates.append(
            {
                "steps": steps,
                "decay": decay,
                "bits": bits,
                "state_relative_l2_mean": float(
                    np.mean([float(row["state_relative_l2"]) for row in group])
                ),
                "state_relative_l2_std": float(
                    np.std([float(row["state_relative_l2"]) for row in group], ddof=1)
                ),
                "trajectory_relative_l2_mean": float(
                    np.mean([float(row["trajectory_relative_l2"]) for row in group])
                ),
                "trajectory_relative_l2_std": float(
                    np.std([float(row["trajectory_relative_l2"]) for row in group], ddof=1)
                ),
            }
        )
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = Config()
    results = [
        recurrent_trial(seed=seed, steps=steps, decay=decay, bits=bits, config=config)
        for seed in config.seeds
        for steps in config.sequence_lengths
        for decay in config.decays
        for bits in config.bits
    ]
    payload = {
        "warning": (
            "Synthetic numerical diagnostic only; not model-quality, latency, or novelty evidence."
        ),
        "config": asdict(config),
        "storage": [storage_summary(config, bits) for bits in config.bits],
        "aggregate": aggregate(results),
        "results": results,
    }
    serialized = json.dumps(payload, indent=2)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
