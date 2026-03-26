from __future__ import annotations

import argparse
from pathlib import Path

from training.runner import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Break C tokenized decoder runner.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Absolute path to a TOML config that starts a new run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_from_config(args.config.resolve())


if __name__ == "__main__":
    main()
