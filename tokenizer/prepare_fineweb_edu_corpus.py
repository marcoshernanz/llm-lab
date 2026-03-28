import argparse
from pathlib import Path

from datasets import load_dataset_builder  # pyright: ignore
from huggingface_hub import HfFileSystem  # pyright: ignore
import pyarrow.parquet as pq  # pyright: ignore


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_MAX_CHARS = 10_000_000
DEFAULT_OUTPUT_PATH = Path("datasets/fineweb_edu/sample10bt_tokenizer_corpus.txt")


def parse_args() -> argparse.Namespace:
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
        default=DEFAULT_MAX_CHARS,
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
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the local tokenizer-training corpus.",
    )
    return parser.parse_args()


def resolve_parquet_paths(dataset_name: str, dataset_config: str, split: str) -> list[str]:
    builder = load_dataset_builder(dataset_name, name=dataset_config)
    data_files = builder.config.data_files
    if data_files is None:
        raise ValueError(f"Dataset config {dataset_config!r} does not expose parquet data files.")

    split_files = data_files.get(split)
    if split_files is None:
        available_splits = ", ".join(sorted(data_files))
        raise ValueError(
            f"Split {split!r} is not available for {dataset_config!r}. "
            f"Available splits: {available_splits}."
        )

    return list(split_files)


def iter_parquet_text(parquet_paths: list[str], *, text_column: str, batch_size: int):
    fs = HfFileSystem()
    for parquet_path in parquet_paths:
        with fs.open(parquet_path, "rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            available_columns = set(parquet_file.schema_arrow.names)
            if text_column not in available_columns:
                available = ", ".join(sorted(available_columns))
                raise ValueError(
                    f"Column {text_column!r} not found in {parquet_path}. "
                    f"Available columns: {available}."
                )

            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[text_column]):
                column = batch.column(0)
                for row_index in range(len(column)):
                    value = column[row_index].as_py()
                    if value is None:
                        continue
                    if not isinstance(value, str):
                        raise ValueError(
                            f"Expected a string {text_column!r} field, got {type(value).__name__}."
                        )
                    yield parquet_path, value.strip()


def main() -> None:
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

            chars_written += len(text_to_write) + 1
            examples_written += 1

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
