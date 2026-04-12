"""Phase 4 baseline: a tiny fixed-configuration PyTorch character LM trainer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn

DATASET_NAME = "roneneldan/TinyStories"
DATASET_CONFIG = None
TRAIN_SPLIT = "train[:20000]"
VALIDATION_SPLIT = "validation[:2000]"
TEXT_COLUMN = "text"
DEVICE = "mps"
SEED = 1337
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 64
SEQUENCE_LENGTH = 128
LEARNING_RATE = 3e-3
TRAIN_STEPS = 2_000
EVAL_INTERVAL = 200
EVAL_BATCHES = 32
SAMPLE_LENGTH = 400


class LanguageModel(nn.Module):
    """Predict the next character from the current character only."""

    def __init__(self, vocab_size: int):
        """Create the embedding table and the two linear layers."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.hidden = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for each token position."""
        x = self.embedding(x)
        x = self.hidden(x)
        x = torch.tanh(x)
        x = self.out(x)
        return x


def load_text(split: str) -> str:
    """Load one text split from Hugging Face and join it into one string."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    parts = [text for text in dataset[TEXT_COLUMN] if text]
    text = "\n".join(parts)
    return text


def build_vocab(train_text: str, validation_text: str) -> tuple[list[str], dict[str, int]]:
    """Build one character vocabulary from the train and validation text."""
    vocab_chars = sorted(set(train_text + validation_text))
    char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
    return vocab_chars, char_to_id


def encode_text(text: str, char_to_id: dict[str, int]) -> torch.Tensor:
    """Turn one text string into a tensor of character ids."""
    return torch.tensor([char_to_id[char] for char in text], dtype=torch.long, device=DEVICE)


def sample_batch(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample contiguous input and target windows from one token stream."""
    max_start = token_ids.size(0) - SEQUENCE_LENGTH - 1
    if max_start < 0:
        raise ValueError("token stream is shorter than one training window")

    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,), device=DEVICE)
    offsets = torch.arange(SEQUENCE_LENGTH, device=DEVICE)
    positions = starts[:, None] + offsets[None, :]
    inputs = token_ids[positions]
    targets = token_ids[positions + 1]
    return inputs, targets


@torch.no_grad()
def estimate_loss(model: LanguageModel, token_ids: torch.Tensor) -> float:
    """Estimate one split loss with a few random batches."""
    losses: list[float] = []
    model.eval()

    for _ in range(EVAL_BATCHES):
        inputs, targets = sample_batch(token_ids)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        losses.append(float(loss.item()))

    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(model: LanguageModel, vocab_chars: list[str], token_ids: torch.Tensor) -> str:
    """Generate a short sample by repeatedly drawing one next character."""
    model.eval()
    token = token_ids[torch.randint(0, token_ids.size(0), (1,), device=DEVICE)].view(1, 1)
    sample = [vocab_chars[int(token.item())]]

    for _ in range(SAMPLE_LENGTH - 1):
        logits = model(token)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        token = torch.multinomial(probs, num_samples=1).view(1, 1)
        sample.append(vocab_chars[int(token.item())])

    model.train()
    return "".join(sample)


def main() -> None:
    """Load the dataset, train the tiny model, and print a short report."""
    torch.manual_seed(SEED)

    train_text = load_text(TRAIN_SPLIT)
    validation_text = load_text(VALIDATION_SPLIT)
    vocab_chars, char_to_id = build_vocab(train_text, validation_text)
    train_token_ids = encode_text(train_text, char_to_id)
    validation_token_ids = encode_text(validation_text, char_to_id)

    model = LanguageModel(len(vocab_chars)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"dataset={DATASET_NAME}")
    print(f"train_split={TRAIN_SPLIT}")
    print(f"validation_split={VALIDATION_SPLIT}")
    print(f"device={DEVICE}")
    print(f"vocab_size={len(vocab_chars)}")
    print(f"train_tokens={train_token_ids.numel()}")
    print(f"validation_tokens={validation_token_ids.numel()}")

    for step in range(TRAIN_STEPS):
        inputs, targets = sample_batch(train_token_ids)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_log = step == 0 or (step + 1) % EVAL_INTERVAL == 0 or step == TRAIN_STEPS - 1
        if should_log:
            train_loss = estimate_loss(model, train_token_ids)
            validation_loss = estimate_loss(model, validation_token_ids)
            print(
                f"step={step + 1} "
                f"batch_loss={loss.item():.4f} "
                f"train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f}"
            )

    sample = sample_text(model, vocab_chars, train_token_ids)
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
