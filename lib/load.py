from pathlib import Path

from tokenizer.bpe import BPEModel


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")
    text = path.read_text(encoding="utf-8")
    if len(text) == 1:
        raise ValueError("Dataset is empty.")
    return text


def load_tokenizer(path: Path) -> BPEModel:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found at {path}.")
    return BPEModel.load(path)
