from pathlib import Path
import tempfile
import unittest

from tokenizer.bpe import BYTE_VOCAB_SIZE
from tokenizer.bpe import train_bpe
from training.config import load_config


class TrainingConfigTests(unittest.TestCase):
    def test_valid_config_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "dataset.txt"
            tokenizer_path = temp_path / "tokenizer.json"
            dataset_path.write_text("hello world\n" * 20, encoding="utf-8")
            train_bpe(dataset_path.read_text(encoding="utf-8"), BYTE_VOCAB_SIZE + 4).save(
                tokenizer_path
            )
            config_path = temp_path / "config.toml"
            config_path.write_text(
                f"""
[run]
seed = 1

[data]
dataset_path = "{dataset_path}"
tokenizer_path = "{tokenizer_path}"
context_tokens = 4
# text_limit = null

[model]
embedding_dim = 8
num_heads = 2
num_decoder_blocks = 1
hidden_dim = 16

[optimizer]
learning_rate = 0.1

[train]
steps = 2
batch_size = 2
eval_batch_size = 2
sample_tokens = 4
""".strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.run.seed, 1)
        self.assertEqual(config.data.context_tokens, 4)
        self.assertIsNone(config.data.text_limit)

    def test_unknown_keys_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "dataset.txt"
            tokenizer_path = temp_path / "tokenizer.json"
            dataset_path.write_text("hello world\n" * 20, encoding="utf-8")
            train_bpe(dataset_path.read_text(encoding="utf-8"), BYTE_VOCAB_SIZE + 4).save(
                tokenizer_path
            )
            config_path = temp_path / "config.toml"
            config_path.write_text(
                f"""
[run]
seed = 1
extra = 1

[data]
dataset_path = "{dataset_path}"
tokenizer_path = "{tokenizer_path}"
context_tokens = 4
# text_limit = null

[model]
embedding_dim = 8
num_heads = 2
num_decoder_blocks = 1
hidden_dim = 16

[optimizer]
learning_rate = 0.1

[train]
steps = 2
batch_size = 2
eval_batch_size = 2
sample_tokens = 4
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Extra inputs are not permitted"):
                _ = load_config(config_path)

    def test_invalid_head_dimension_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "dataset.txt"
            tokenizer_path = temp_path / "tokenizer.json"
            dataset_path.write_text("hello world\n" * 20, encoding="utf-8")
            train_bpe(dataset_path.read_text(encoding="utf-8"), BYTE_VOCAB_SIZE + 4).save(
                tokenizer_path
            )
            config_path = temp_path / "config.toml"
            config_path.write_text(
                f"""
[run]
seed = 1

[data]
dataset_path = "{dataset_path}"
tokenizer_path = "{tokenizer_path}"
context_tokens = 4
# text_limit = null

[model]
embedding_dim = 7
num_heads = 2
num_decoder_blocks = 1
hidden_dim = 16

[optimizer]
learning_rate = 0.1

[train]
steps = 2
batch_size = 2
eval_batch_size = 2
sample_tokens = 4
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "divisible"):
                _ = load_config(config_path)

    def test_invalid_seed_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "dataset.txt"
            tokenizer_path = temp_path / "tokenizer.json"
            dataset_path.write_text("hello world\n" * 20, encoding="utf-8")
            train_bpe(dataset_path.read_text(encoding="utf-8"), BYTE_VOCAB_SIZE + 4).save(
                tokenizer_path
            )
            config_path = temp_path / "config.toml"
            config_path.write_text(
                f"""
[run]
seed = 0

[data]
dataset_path = "{dataset_path}"
tokenizer_path = "{tokenizer_path}"
context_tokens = 4
# text_limit = null

[model]
embedding_dim = 8
num_heads = 2
num_decoder_blocks = 1
hidden_dim = 16

[optimizer]
learning_rate = 0.1

[train]
steps = 2
batch_size = 2
eval_batch_size = 2
sample_tokens = 4
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "greater than 0"):
                _ = load_config(config_path)


if __name__ == "__main__":
    unittest.main()
