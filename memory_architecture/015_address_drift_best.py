"""Memory architecture experiment 015: best address-drift control run."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


ADDRESS_UPDATE_SCALE = 0.075
SEED = 1337
TRAIN_STEPS = 4_000
EVAL_INTERVAL = 200
EVAL_BATCHES = 32


def main() -> None:
    """Run the best-performing M-015 address-drift setting."""
    experiment = _load_m014()
    experiment.DEVICE = _choose_device()
    experiment.SEED = int(os.environ.get("LLM_LAB_SEED", str(SEED)))
    experiment.ADDRESS_UPDATE_SCALE = float(
        os.environ.get("LLM_LAB_ADDRESS_UPDATE_SCALE", str(ADDRESS_UPDATE_SCALE))
    )
    experiment.TRAIN_STEPS = int(os.environ.get("LLM_LAB_TRAIN_STEPS", str(TRAIN_STEPS)))
    experiment.EVAL_INTERVAL = int(os.environ.get("LLM_LAB_EVAL_INTERVAL", str(EVAL_INTERVAL)))
    experiment.EVAL_BATCHES = int(os.environ.get("LLM_LAB_EVAL_BATCHES", str(EVAL_BATCHES)))
    experiment.__file__ = __file__

    print("experiment_name=015_address_drift_best")
    print(f"device={experiment.DEVICE}")
    print(f"seed={experiment.SEED}")
    print(f"train_steps={experiment.TRAIN_STEPS}")
    print(f"eval_interval={experiment.EVAL_INTERVAL}")
    print(f"eval_batches={experiment.EVAL_BATCHES}")
    print(f"address_update_scale={experiment.ADDRESS_UPDATE_SCALE}")

    experiment.main()


def _load_m014():
    """Load experiment 014 despite its numeric filename."""
    module_path = Path(__file__).with_name("014_bounded_address_drift.py")
    spec = importlib.util.spec_from_file_location("m014_bounded_address_drift", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _choose_device() -> str:
    """Pick the requested accelerator, then CUDA, then MPS, then CPU."""
    requested_device = os.environ.get("LLM_LAB_DEVICE")
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
