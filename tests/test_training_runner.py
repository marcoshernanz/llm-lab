from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tokenizer.bpe import BYTE_VOCAB_SIZE
from tokenizer.bpe import train_bpe


class TrainingRunnerTests(unittest.TestCase):
    def test_cli_run_creates_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = _write_tiny_training_fixture(temp_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent.parent / "train.py"),
                    "--config",
                    str(config_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            run_dir = _extract_run_dir(result.stdout)
            self.assertTrue((run_dir / "config.toml").exists())
            self.assertTrue((run_dir / "run_metadata.json").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "loss_curve.svg").exists())
            self.assertTrue((run_dir / "samples" / "step_000000.txt").exists())
            self.assertTrue((run_dir / "samples" / "step_000003.txt").exists())

            metrics_rows = _read_metrics(run_dir / "metrics.csv")
            self.assertEqual(len(metrics_rows), 3)
            self.assertEqual(metrics_rows[-1]["step"], "2")
            self.assertNotEqual(metrics_rows[-1]["validation_loss"], "")

            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["recipe_name"], "tokenized_decoder_jax")
            self.assertIn("train_summary", metadata)


def _write_tiny_training_fixture(root: Path, *, steps: int = 3) -> Path:
    dataset_path = root / "dataset.txt"
    tokenizer_path = root / "tokenizer.json"
    dataset_path.write_text(
        "To be, or not to be, that is the question.\n" * 50,
        encoding="utf-8",
    )
    train_bpe(dataset_path.read_text(encoding="utf-8"), BYTE_VOCAB_SIZE + 8).save(tokenizer_path)
    config_path = root / "config.toml"
    config_path.write_text(
        f"""
[run]
seed = 7

[data]
dataset_path = "{dataset_path}"
tokenizer_path = "{tokenizer_path}"
context_tokens = 8
# text_limit = null

[model]
embedding_dim = 8
num_heads = 2
num_decoder_blocks = 1
hidden_dim = 16

[optimizer]
learning_rate = 0.05

[train]
steps = {steps}
batch_size = 2
eval_batch_size = 4
sample_tokens = 12
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _extract_run_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("run_dir="):
            return Path(line.removeprefix("run_dir="))
    raise AssertionError(f"Could not find run_dir in stdout:\n{stdout}")


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
