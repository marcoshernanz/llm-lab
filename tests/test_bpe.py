import json
from pathlib import Path

import pytest

from tokenizer.bpe import BYTE_VOCAB_SIZE
from tokenizer.bpe import BPEModel
from tokenizer.bpe import TOKENIZER_ARTIFACT_VERSION
from tokenizer.bpe import split_text
from tokenizer.bpe import train_bpe


def test_split_text_preserves_round_trip() -> None:
    text = "hi\n\na  b\n"
    rebuilt = b"".join(split_text(text)).decode("utf-8")
    assert rebuilt == text


def test_train_bpe_raises_for_vocab_below_byte_range() -> None:
    with pytest.raises(
        ValueError,
        match=rf"vocab_size must be at least {BYTE_VOCAB_SIZE} for byte-level BPE\.",
    ):
        _ = train_bpe("hello", BYTE_VOCAB_SIZE - 1)


def test_train_bpe_empty_text_returns_base_vocab() -> None:
    model = train_bpe("", BYTE_VOCAB_SIZE + 10)

    assert model.vocab_size == BYTE_VOCAB_SIZE
    assert model.merges == ()
    assert model.encode("") == []
    assert model.decode([]) == ""


def test_encode_decode_round_trip() -> None:
    text = "banana bandana\n"
    model = train_bpe(text, BYTE_VOCAB_SIZE + 4)

    token_ids = model.encode(text)

    assert any(token_id >= BYTE_VOCAB_SIZE for token_id in token_ids)
    assert model.decode(token_ids) == text


def test_training_tie_break_is_deterministic() -> None:
    model = train_bpe("ab ac", BYTE_VOCAB_SIZE + 1)

    pair, token_id = model.merges[0]

    assert pair == (32, 97)
    assert model.vocab[token_id] == b" a"


def test_save_load_preserves_tokenizer_behavior(tmp_path: Path) -> None:
    text = "banana bandana\n"
    model = train_bpe(text, BYTE_VOCAB_SIZE + 4)

    artifact_path = tmp_path / "tokenizer.json"
    model.save(artifact_path)
    loaded_model = BPEModel.load(artifact_path)

    assert loaded_model.split_pattern == model.split_pattern
    assert loaded_model.merges == model.merges
    assert loaded_model.vocab_size == model.vocab_size
    assert loaded_model.encode(text) == model.encode(text)
    assert loaded_model.decode(model.encode(text)) == text


def test_save_writes_minimal_artifact_schema(tmp_path: Path) -> None:
    model = train_bpe("banana bandana\n", BYTE_VOCAB_SIZE + 2)

    artifact_path = tmp_path / "tokenizer.json"
    model.save(artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["version"] == TOKENIZER_ARTIFACT_VERSION
    assert payload["split_pattern"] == model.split_pattern
    assert payload["merge_pairs"] == [[97, 110], [98, 256]]
