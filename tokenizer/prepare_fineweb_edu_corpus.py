import argparse
from pathlib import Path
from typing import Any


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_MAX_CHARS = 10_000_000
DEFAULT_OUTPUT_PATH = Path("datasets/fineweb_edu/sample10bt_tokenizer_corpus.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream FineWeb-Edu and write a capped local text corpus for tokenizer training."
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
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to stream.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Column containing the text payload.",
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
        help="Optional maximum number of streamed examples to consume.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the local tokenizer-training corpus.",
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
    split: str,
    text_column: str,
):
    from datasets import load_dataset  # pyright: ignore

    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
    )
    for record in dataset:
        yield get_text(record, text_column).strip()


def main() -> None:
    args = parse_args()
    if args.max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    chars_written = 0
    examples_written = 0

    with args.output_path.open("w", encoding="utf-8") as handle:
        for text in stream_fineweb_text(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            text_column=args.text_column,
        ):
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
    print(f"examples_written={examples_written}")
    print(f"chars_written={chars_written}")


if __name__ == "__main__":
    main()
