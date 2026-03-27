import optax  # pyright: ignore
from flax import nnx

import jax
import jax.nn as jnn
import jax.numpy as jnp

from pathlib import Path

from lib.data import build_examples
from lib.data import build_token_splits
from lib.data import load_text
from lib.data import load_tokenizer
from lib.eval import evaluate_split
from lib.timer import Timer
from models.transformer import DecoderOnlyTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "datasets" / "tinyshakespeare.txt"
TOKENIZER_PATH = ROOT_DIR / "artifacts" / "tokenizers" / "tinyshakespeare_bpe_512.json"

SEED = 0
TRAIN_SPLIT = 0.8
EVAL_BATCH_SIZE = 64
BATCH_SIZE = 16
LEARNING_RATE = 0.02
TRAIN_STEPS = 50_000
TRAIN_CHUNK_LENGTH = 1000
if TRAIN_STEPS % TRAIN_CHUNK_LENGTH != 0:
    raise ValueError("TRAIN_STEPS must be divisible by TRAIN_CHUNK_LENGTH")

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_DECODER_BLOCKS = 4
HIDDEN_DIM = 128
CONTEXT_LENGTH = 64


def loss_fn(
    model: DecoderOnlyTransformer,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@nnx.jit
def train_step(
    model: DecoderOnlyTransformer,
    optimizer: nnx.Optimizer[DecoderOnlyTransformer],
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    loss, grads = nnx.value_and_grad(loss_fn)(model, input_ids, target_ids)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def evaluate_batch_loss(
    model: DecoderOnlyTransformer,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    return loss_fn(model, input_ids, target_ids)


def train_chunk(
    model: DecoderOnlyTransformer,
    optimizer: nnx.Optimizer[DecoderOnlyTransformer],
    tokens: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    loss = jnp.array(jnp.nan, dtype=jnp.float32)
    for _ in range(TRAIN_CHUNK_LENGTH):
        rng, batch_rng = jax.random.split(rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(BATCH_SIZE,),
            minval=0,
            maxval=tokens.shape[0] - CONTEXT_LENGTH,
        )
        input_ids, target_ids = build_examples(tokens, start_positions, CONTEXT_LENGTH)
        loss = train_step(model, optimizer, input_ids, target_ids)
    return loss, rng


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

    rng = jax.random.key(SEED)
    for step in range(0, TRAIN_STEPS, TRAIN_CHUNK_LENGTH):
        loss, rng = train_chunk(model, optimizer, train_tokens, rng)
        print(f"step={step} loss={loss:.6f}")

    train_seconds = timer.stop("train")
    train_loss = evaluate_split(
        train_tokens,
        model,
        evaluate_batch_loss,
        CONTEXT_LENGTH,
        EVAL_BATCH_SIZE,
    )
    validation_loss = evaluate_split(
        validation_tokens,
        model,
        evaluate_batch_loss,
        CONTEXT_LENGTH,
        EVAL_BATCH_SIZE,
    )
    total_seconds = timer.stop("total")

    print(f"train_loss={train_loss:.6f}")
    print(f"validation_loss={validation_loss:.6f}")
    print(f"train_seconds={train_seconds:.3f}")
    print(f"steps_per_second={TRAIN_STEPS / train_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")


if __name__ == "__main__":
    main()
