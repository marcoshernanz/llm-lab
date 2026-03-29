"""Create a small local FineWeb-Edu text corpus for tokenizer training."""

import argparse
from pathlib import Path
from tokenizer.fineweb_edu import DEFAULT_BATCH_SIZE
from tokenizer.fineweb_edu import DEFAULT_DATASET_CONFIG
from tokenizer.fineweb_edu import DEFAULT_DATASET_NAME
from tokenizer.fineweb_edu import DEFAULT_SPLIT
from tokenizer.fineweb_edu import DEFAULT_TEXT_COLUMN
from tokenizer.fineweb_edu import iter_parquet_text
from tokenizer.fineweb_edu import resolve_parquet_paths

LOG_EVERY_CHARS = 1_000_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for corpus preparation."""
    parser = argparse.ArgumentParser(
        description="Read FineWeb-Edu parquet shards and write a capped local text corpus for tokenizer training."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to inspect.",
    )
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
        help="Hugging Face dataset config or subset name.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to read.",
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
        "--max-chars",
        type=int,
        required=True,
        help="Maximum number of characters to write to the local corpus.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional maximum number of examples to consume.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to the local tokenizer-training corpus.",
    )
    return parser.parse_args()


def main() -> None:
    """Read FineWeb-Edu text and write a capped local training corpus."""
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")

    parquet_paths = resolve_parquet_paths(args.dataset_name, args.dataset_config, args.split)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    chars_written = 0
    examples_written = 0
    shards_touched = 0
    current_shard: str | None = None
    next_char_log = LOG_EVERY_CHARS

    with args.output_path.open("w", encoding="utf-8") as handle:
        for parquet_path, text in iter_parquet_text(
            parquet_paths,
            text_column=args.text_column,
            batch_size=args.batch_size,
        ):
            if parquet_path != current_shard:
                current_shard = parquet_path
                shards_touched += 1

            if not text:
                continue

            remaining_chars = args.max_chars - chars_written
            if remaining_chars <= 0:
                break

            text_to_write = text[:remaining_chars]
            handle.write(text_to_write)
            handle.write("\n")

            written_chars = len(text_to_write)
            chars_written += written_chars + 1
            examples_written += 1

            if chars_written >= next_char_log:
                next_char_log += LOG_EVERY_CHARS
                print(f"chars_written={chars_written}")

            if chars_written >= args.max_chars:
                break
            if args.max_examples is not None and examples_written >= args.max_examples:
                break

    print(f"dataset={args.dataset_name}")
    print(f"dataset_config={args.dataset_config}")
    print(f"split={args.split}")
    print(f"output_path={args.output_path}")
    print(f"parquet_files={len(parquet_paths)}")
    print(f"shards_touched={shards_touched}")
    print(f"examples_written={examples_written}")
    print(f"chars_written={chars_written}")
    if current_shard is not None:
        print(f"last_shard={current_shard}")


if __name__ == "__main__":
    main()
