# %%

import optax  # pyright: ignore
from flax import nnx

from pathlib import Path

from lib.timer import Timer
from lib.utils import load_text, load_tokenizer, build_token_splits
from models.transformer import DecoderOnlyTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "datasets" / "tinyshakespeare.txt"
TOKENIZER_PATH = ROOT_DIR / "artifacts" / "tokenizers" / "tinyshakespeare_bpe_512.json"

SEED = 0
TRAIN_SPLIT = 0.8
EVAL_BATCH_SIZE = 64
SAMPLE_LENGTH = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.02
TRAIN_STEPS = 50_000
LOG_INTERVAL = 5000

EMBEDDING_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
NUM_DECODER_BLOCKS = 4
HIDDEN_DIM = 128
CONTEXT_LENGTH = 64


def main():
    timer = Timer()
    timer.start("total")
    rngs = nnx.Rngs(SEED)
    text = load_text(DATA_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    train_tokens, validation_tokens = build_token_splits(text, tokenizer, TRAIN_SPLIT)

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_decoder_blocks=NUM_DECODER_BLOCKS,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.sgd(LEARNING_RATE), wrt=nnx.Param)
    timer.start("train")


if __name__ == "__main__":
    main()
