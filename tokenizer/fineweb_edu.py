from typing import Iterator

from datasets import load_dataset_builder  # pyright: ignore
from huggingface_hub import HfFileSystem  # pyright: ignore
import pyarrow.parquet as pq  # pyright: ignore


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_BATCH_SIZE = 1024


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


def iter_parquet_text(
    parquet_paths: list[str], *, text_column: str, batch_size: int
) -> Iterator[tuple[str, str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

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
