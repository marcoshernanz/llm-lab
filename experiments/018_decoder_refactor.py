# %%

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "datasets" / "tinyshakespeare.txt"
TOKENIZER_PATH = ROOT_DIR / "artifacts" / "tokenizers" / "tinyshakespeare_bpe_512.json"

SEED = 0
EMBEDDING_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
NUM_DECODER_BLOCKS = 4
HIDDEN_DIM = 128
CONTEXT_LENGTH = 64
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
SAMPLE_LENGTH = 100
LEARNING_RATE = 0.02
TRAIN_STEPS = 50_000
LOG_INTERVAL = 5000
