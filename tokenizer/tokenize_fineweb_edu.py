import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tokenizer.bpe import BPEModel


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_SOURCE_SPLIT = "train"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_OUTPUT_DIR = Path("datasets/fineweb_edu/sample10bt_bpe_16384")
DEFAULT_SHARD_TOKENS = 10_000_000
DEFAULT_VALIDATION_FRACTION = 0.01
DEFAULT_DOCUMENT_SEPARATOR = "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream FineWeb-Edu, tokenize it, and write train/validation token shards."
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
    return parser.parse_args()


def get_text(record: dict[str, Any], text_column: str) -> str:
    value = record.get(text_column)
    if not isinstance(value, str):
        raise ValueError(f"Expected a string {text_column!r} field, got {type(value).__name__}.")
    return value


def stream_fineweb_text(
    *,
    dataset_name: str,
    dataset_config: str,
    source_split: str,
    text_column: str,
):
    from datasets import load_dataset  # pyright: ignore

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=source_split,
        streaming=True,
    )
    for record in dataset:
        yield get_text(record, text_column).strip()


def choose_split(text: str, validation_fraction: float) -> str:
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
) -> None:
    shard_path = output_dir / f"{split}_{shard_index:05d}.npy"
    np.save(shard_path, np.asarray(token_ids, dtype=np.int32))


def flush_completed_shards(
    *,
    output_dir: Path,
    split: str,
    shard_tokens: int,
    buffer: list[int],
    next_shard_index: int,
) -> tuple[list[int], int, int]:
    tokens_written = 0
    while len(buffer) >= shard_tokens:
        write_shard(output_dir, split, next_shard_index, buffer[:shard_tokens])
        tokens_written += shard_tokens
        buffer = buffer[shard_tokens:]
        next_shard_index += 1
    return buffer, next_shard_index, tokens_written


def write_remainder_shard(
    *,
    output_dir: Path,
    split: str,
    buffer: list[int],
    next_shard_index: int,
) -> tuple[int, int]:
    if not buffer:
        return next_shard_index, 0
    write_shard(output_dir, split, next_shard_index, buffer)
    return next_shard_index + 1, len(buffer)


def main() -> None:
    args = parse_args()
    if args.shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if args.max_documents is not None and args.max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")

    tokenizer = BPEModel.load(args.tokenizer_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_buffers = {"train": [], "validation": []}
    next_shard_indices = {"train": 0, "validation": 0}
    token_totals = {"train": 0, "validation": 0}
    document_totals = {"train": 0, "validation": 0}

    for document_index, text in enumerate(
        stream_fineweb_text(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            source_split=args.source_split,
            text_column=args.text_column,
        ),
        start=1,
    ):
        if not text:
            continue

        split = choose_split(text, args.validation_fraction)
        encoded_document = tokenizer.encode(text + args.document_separator)
        if not encoded_document:
            continue

        split_buffers[split].extend(encoded_document)
        document_totals[split] += 1

        (
            split_buffers[split],
            next_shard_indices[split],
            written_tokens,
        ) = flush_completed_shards(
            output_dir=args.output_dir,
            split=split,
            shard_tokens=args.shard_tokens,
            buffer=split_buffers[split],
            next_shard_index=next_shard_indices[split],
        )
        token_totals[split] += written_tokens

        if args.max_documents is not None and document_index >= args.max_documents:
            break

    for split in ("train", "validation"):
        next_shard_indices[split], written_tokens = write_remainder_shard(
            output_dir=args.output_dir,
            split=split,
            buffer=split_buffers[split],
            next_shard_index=next_shard_indices[split],
        )
        token_totals[split] += written_tokens

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "source_split": args.source_split,
        "text_column": args.text_column,
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "output_dir": str(args.output_dir),
        "shard_tokens": args.shard_tokens,
        "validation_fraction": args.validation_fraction,
        "document_separator": args.document_separator,
        "documents": document_totals,
        "tokens": token_totals,
        "shards": next_shard_indices,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"output_dir={args.output_dir}")
    print(f"metadata_path={metadata_path}")
    print(f"train_documents={document_totals['train']}")
    print(f"validation_documents={document_totals['validation']}")
    print(f"train_tokens={token_totals['train']}")
    print(f"validation_tokens={token_totals['validation']}")
    print(f"train_shards={next_shard_indices['train']}")
    print(f"validation_shards={next_shard_indices['validation']}")


if __name__ == "__main__":
    main()
