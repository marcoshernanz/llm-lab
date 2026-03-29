"""Tokenize FineWeb-Edu documents into train and validation shard files."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from tokenizer.bpe import BPEModel
from tokenizer.fineweb_edu import DEFAULT_BATCH_SIZE
from tokenizer.fineweb_edu import DEFAULT_DATASET_CONFIG
from tokenizer.fineweb_edu import DEFAULT_DATASET_NAME
from tokenizer.fineweb_edu import DEFAULT_SPLIT
from tokenizer.fineweb_edu import DEFAULT_TEXT_COLUMN
from tokenizer.fineweb_edu import iter_parquet_text
from tokenizer.fineweb_edu import resolve_parquet_paths


DEFAULT_SOURCE_SPLIT = DEFAULT_SPLIT
DEFAULT_OUTPUT_DIR = Path("datasets/fineweb_edu/sample10bt_bpe_16384")
DEFAULT_SHARD_TOKENS = 10_000_000
DEFAULT_VALIDATION_FRACTION = 0.01
DEFAULT_DOCUMENT_SEPARATOR = "\n"
LOG_EVERY_DOCUMENTS = 1_000


def choose_token_dtype(vocab_size: int) -> np.dtype:
    """Choose a compact dtype that can represent the tokenizer vocabulary."""
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.int32)


@dataclass
class SplitWriter:
    """Accumulate tokenized documents and flush them into shard files."""

    output_dir: Path
    split: str
    shard_tokens: int
    token_dtype: np.dtype
    buffer: list[int]
    next_shard_index: int = 0
    documents: int = 0
    tokens: int = 0

    def append(self, token_ids: list[int]) -> None:
        """Append one document and write any full shards it creates."""
        self.buffer.extend(token_ids)
        self.documents += 1
        while len(self.buffer) >= self.shard_tokens:
            write_shard(
                self.output_dir,
                self.split,
                self.next_shard_index,
                self.buffer[: self.shard_tokens],
                self.token_dtype,
            )
            self.tokens += self.shard_tokens
            self.buffer = self.buffer[self.shard_tokens :]
            self.next_shard_index += 1

    def finalize(self) -> None:
        """Write the final partial shard, if any tokens remain."""
        if not self.buffer:
            return
        write_shard(self.output_dir, self.split, self.next_shard_index, self.buffer, self.token_dtype)
        self.tokens += len(self.buffer)
        self.next_shard_index += 1
        self.buffer = []


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for token shard generation."""
    parser = argparse.ArgumentParser(
        description="Read FineWeb-Edu parquet shards, tokenize them, and write train/validation token shards."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to stream.",
    )
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
        help="Hugging Face dataset config or subset name.",
    )
    parser.add_argument(
        "--source-split",
        default=DEFAULT_SOURCE_SPLIT,
        help="Dataset split to stream from Hugging Face.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Column containing the text payload.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to read from parquet at a time.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="Path to the tokenizer artifact used for encoding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where token shards and metadata will be written.",
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=DEFAULT_SHARD_TOKENS,
        help="Target number of tokens per shard file.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Fraction of documents reserved for validation.",
    )
    parser.add_argument(
        "--document-separator",
        default=DEFAULT_DOCUMENT_SEPARATOR,
        help="String appended after each document before tokenization.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional maximum number of documents to stream for smoke tests.",
    )
    parser.add_argument(
        "--max-train-shards",
        type=int,
        default=None,
        help="Optional cap on the number of completed train shards to write.",
    )
    return parser.parse_args()


def choose_split(text: str, validation_fraction: float) -> str:
    """Deterministically assign a document to train or validation."""
    if validation_fraction == 0.0:
        return "train"

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    sample = int.from_bytes(digest, byteorder="big") / 2**64
    return "validation" if sample < validation_fraction else "train"


def write_shard(
    output_dir: Path,
    split: str,
    shard_index: int,
    token_ids: list[int],
    token_dtype: np.dtype,
) -> None:
    """Write one token shard to disk as a NumPy array."""
    shard_path = output_dir / f"{split}_{shard_index:05d}.npy"
    np.save(shard_path, np.asarray(token_ids, dtype=token_dtype))


def main() -> None:
    """Stream FineWeb-Edu, tokenize it, and save token shard metadata."""
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if args.max_documents is not None and args.max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")
    if args.max_train_shards is not None and args.max_train_shards <= 0:
        raise ValueError("max_train_shards must be positive when provided")

    tokenizer = BPEModel.load(args.tokenizer_path)
    parquet_paths = resolve_parquet_paths(args.dataset_name, args.dataset_config, args.source_split)
    token_dtype = choose_token_dtype(tokenizer.vocab_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_writers = {
        "train": SplitWriter(args.output_dir, "train", args.shard_tokens, token_dtype, []),
        "validation": SplitWriter(args.output_dir, "validation", args.shard_tokens, token_dtype, []),
    }

    shards_touched = 0
    current_shard: str | None = None

    for document_index, (parquet_path, text) in enumerate(
        iter_parquet_text(parquet_paths, text_column=args.text_column, batch_size=args.batch_size),
        start=1,
    ):
        if parquet_path != current_shard:
            current_shard = parquet_path
            shards_touched += 1

        if not text:
            continue

        split = choose_split(text, args.validation_fraction)
        encoded_document = tokenizer.encode(text + args.document_separator)
        if not encoded_document:
            continue

        split_writers[split].append(encoded_document)
        if document_index % LOG_EVERY_DOCUMENTS == 0:
            print(f"documents={document_index}")

        if (
            args.max_train_shards is not None
            and split_writers["train"].next_shard_index >= args.max_train_shards
        ):
            split_writers["train"].buffer = []
            break

        if args.max_documents is not None and document_index >= args.max_documents:
            break

    for split in ("train", "validation"):
        split_writers[split].finalize()

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "source_split": args.source_split,
        "text_column": args.text_column,
        "batch_size": args.batch_size,
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "token_dtype": token_dtype.name,
        "output_dir": str(args.output_dir),
        "shard_tokens": args.shard_tokens,
        "validation_fraction": args.validation_fraction,
        "document_separator": args.document_separator,
        "max_train_shards": args.max_train_shards,
        "parquet_files": len(parquet_paths),
        "shards_touched": shards_touched,
        "documents": {split: split_writers[split].documents for split in split_writers},
        "tokens": {split: split_writers[split].tokens for split in split_writers},
        "shards": {split: split_writers[split].next_shard_index for split in split_writers},
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"output_dir={args.output_dir}")
    print(f"metadata_path={metadata_path}")
    print(f"parquet_files={len(parquet_paths)}")
    print(f"shards_touched={shards_touched}")
    print(f"train_documents={split_writers['train'].documents}")
    print(f"validation_documents={split_writers['validation'].documents}")
    print(f"train_tokens={split_writers['train'].tokens}")
    print(f"validation_tokens={split_writers['validation'].tokens}")
    print(f"train_shards={split_writers['train'].next_shard_index}")
    print(f"validation_shards={split_writers['validation'].next_shard_index}")
    if current_shard is not None:
        print(f"last_shard={current_shard}")


if __name__ == "__main__":
    main()
