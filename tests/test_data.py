"""Test the small data-loading helpers used by the learning experiments."""

import json
from pathlib import Path

import numpy as np
import pytest

from lib.data import list_token_shards
from lib.data import load_token_shard
from lib.data import load_token_shard_metadata
from lib.data import load_token_split_from_shards


def test_list_token_shards_returns_sorted_matches(tmp_path: Path) -> None:
    """Ensure shard discovery keeps lexicographic shard order."""
    (tmp_path / "train_00001.npy").write_bytes(b"")
    (tmp_path / "train_00000.npy").write_bytes(b"")

    shard_paths = list_token_shards(tmp_path, "train")

    assert [path.name for path in shard_paths] == ["train_00000.npy", "train_00001.npy"]


def test_list_token_shards_raises_when_missing(tmp_path: Path) -> None:
    """Ensure missing shard splits fail with a clear error."""
    with pytest.raises(FileNotFoundError, match="No 'validation' token shards found"):
        _ = list_token_shards(tmp_path, "validation")


def test_load_token_shard_reads_int32_vector(tmp_path: Path) -> None:
    """Ensure shard loading normalizes tokens to JAX int32 arrays."""
    shard_path = tmp_path / "train_00000.npy"
    np.save(shard_path, np.asarray([1, 2, 3], dtype=np.int64))

    token_ids = load_token_shard(shard_path)

    assert token_ids.shape == (3,)
    assert token_ids.dtype.name == "int32"
    assert token_ids.tolist() == [1, 2, 3]


def test_load_token_shard_supports_mmap(tmp_path: Path) -> None:
    """Ensure optional memory mapping still returns the expected JAX tokens."""
    shard_path = tmp_path / "train_00000.npy"
    np.save(shard_path, np.asarray([1, 2, 3], dtype=np.uint16))

    token_ids = load_token_shard(shard_path, mmap=True)

    assert token_ids.shape == (3,)
    assert token_ids.dtype.name == "int32"
    assert token_ids.tolist() == [1, 2, 3]


def test_load_token_shard_metadata_reads_json(tmp_path: Path) -> None:
    """Ensure shard metadata is read back as plain JSON."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"train_tokens": 12}), encoding="utf-8")

    metadata = load_token_shard_metadata(tmp_path)

    assert metadata == {"train_tokens": 12}


def test_load_token_split_from_shards_concatenates_sorted_shards(tmp_path: Path) -> None:
    """Ensure split loading concatenates shards in order."""
    np.save(tmp_path / "train_00001.npy", np.asarray([3, 4], dtype=np.int32))
    np.save(tmp_path / "train_00000.npy", np.asarray([1, 2], dtype=np.int32))

    token_ids = load_token_split_from_shards(tmp_path, "train")

    assert token_ids.tolist() == [1, 2, 3, 4]


def test_load_token_split_from_shards_respects_max_shards(tmp_path: Path) -> None:
    """Ensure split loading can cap the number of concatenated shards."""
    np.save(tmp_path / "validation_00000.npy", np.asarray([5, 6], dtype=np.int32))
    np.save(tmp_path / "validation_00001.npy", np.asarray([7, 8], dtype=np.int32))

    token_ids = load_token_split_from_shards(tmp_path, "validation", max_shards=1)

    assert token_ids.tolist() == [5, 6]
