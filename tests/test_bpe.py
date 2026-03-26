import json
from pathlib import Path
import tempfile
import unittest

from tokenizer.bpe import BYTE_VOCAB_SIZE
from tokenizer.bpe import BPEModel
from tokenizer.bpe import TOKENIZER_ARTIFACT_VERSION
from tokenizer.bpe import split_text
from tokenizer.bpe import train_bpe


class BPETests(unittest.TestCase):
    def test_split_text_preserves_round_trip(self) -> None:
        text = "hi\n\na  b\n"
        rebuilt = b"".join(split_text(text)).decode("utf-8")
        self.assertEqual(rebuilt, text)

    def test_train_bpe_raises_for_vocab_below_byte_range(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            rf"vocab_size must be at least {BYTE_VOCAB_SIZE} for byte-level BPE\.",
        ):
            _ = train_bpe("hello", BYTE_VOCAB_SIZE - 1)

    def test_train_bpe_empty_text_returns_base_vocab(self) -> None:
        model = train_bpe("", BYTE_VOCAB_SIZE + 10)
        self.assertEqual(model.vocab_size, BYTE_VOCAB_SIZE)
        self.assertEqual(model.merges, ())
        self.assertEqual(model.encode(""), [])
        self.assertEqual(model.decode([]), "")

    def test_encode_decode_round_trip(self) -> None:
        text = "banana bandana\n"
        model = train_bpe(text, BYTE_VOCAB_SIZE + 4)

        token_ids = model.encode(text)

        self.assertTrue(any(token_id >= BYTE_VOCAB_SIZE for token_id in token_ids))
        self.assertEqual(model.decode(token_ids), text)

    def test_training_tie_break_is_deterministic(self) -> None:
        model = train_bpe("ab ac", BYTE_VOCAB_SIZE + 1)

        pair, token_id = model.merges[0]

        self.assertEqual(pair, (32, 97))
        self.assertEqual(model.vocab[token_id], b" a")

    def test_save_load_preserves_tokenizer_behavior(self) -> None:
        text = "banana bandana\n"
        model = train_bpe(text, BYTE_VOCAB_SIZE + 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "tokenizer.json"
            model.save(artifact_path)
            loaded_model = BPEModel.load(artifact_path)

        self.assertEqual(loaded_model.split_pattern, model.split_pattern)
        self.assertEqual(loaded_model.merges, model.merges)
        self.assertEqual(loaded_model.vocab_size, model.vocab_size)
        self.assertEqual(loaded_model.encode(text), model.encode(text))
        self.assertEqual(loaded_model.decode(model.encode(text)), text)

    def test_save_writes_minimal_artifact_schema(self) -> None:
        model = train_bpe("banana bandana\n", BYTE_VOCAB_SIZE + 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "tokenizer.json"
            model.save(artifact_path)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], TOKENIZER_ARTIFACT_VERSION)
        self.assertEqual(payload["split_pattern"], model.split_pattern)
        self.assertEqual(payload["merge_pairs"], [[97, 110], [98, 256]])


if __name__ == "__main__":
    unittest.main()
