import optax  # pyright: ignore
from flax import nnx

import jax
import jax.nn as jnn
import jax.numpy as jnp

from pathlib import Path

from lib.data import build_examples
from lib.data import list_token_shards
from lib.data import load_token_shard
from lib.data import load_token_shard_metadata
from lib.data import load_tokenizer
from lib.eval import evaluate_positions, evaluate_split, sample_evaluation_positions
from lib.plotting import LossTracker
from lib.timer import Timer
from models.transformer import DecoderOnlyTransformer
from tokenizer.bpe import BPEModel

ROOT_DIR = Path(__file__).resolve().parent.parent
TOKEN_SHARD_ROOT = ROOT_DIR / "datasets" / "fineweb_edu" / "sample10bt_bpe_16384"
TOKENIZER_PATH = ROOT_DIR / "artifacts" / "tokenizers" / "fineweb_edu_sample10bt_bpe_16384.json"

SEED = 0
TRAIN_SHARD_INDEX = 0
VALIDATION_SHARD_INDEX = 0
EVAL_BATCH_SIZE = 64
BATCH_SIZE = 16
LEARNING_RATE = 0.02
TRAIN_STEPS = 50_000
TRAIN_CHUNK_LENGTH = 1000
VALIDATION_SUBSET_EXAMPLES = 1024
SAMPLE_TOKENS = 100
if TRAIN_STEPS % TRAIN_CHUNK_LENGTH != 0:
    raise ValueError("TRAIN_STEPS must be divisible by TRAIN_CHUNK_LENGTH")

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_DECODER_BLOCKS = 4
HIDDEN_DIM = 128
CONTEXT_LENGTH = 64


def load_experiment_split(root_dir: Path, split: str, shard_index: int) -> tuple[Path, jax.Array]:
    shard_paths = list_token_shards(root_dir, split)
    if shard_index < 0 or shard_index >= len(shard_paths):
        raise ValueError(
            f"{split} shard index {shard_index} is out of range for {root_dir}. "
            f"Available {split} shards: {len(shard_paths)}."
        )
    shard_path = shard_paths[shard_index]
    return shard_path, load_token_shard(shard_path)


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
    total_loss = jnp.array(0.0, dtype=jnp.float32)
    for _ in range(TRAIN_CHUNK_LENGTH):
        rng, batch_rng = jax.random.split(rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(BATCH_SIZE,),
            minval=0,
            maxval=tokens.shape[0] - CONTEXT_LENGTH,
        )
        input_ids, target_ids = build_examples(tokens, start_positions, CONTEXT_LENGTH)
        total_loss = total_loss + train_step(model, optimizer, input_ids, target_ids)
    return total_loss / TRAIN_CHUNK_LENGTH, rng


def generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: BPEModel,
    seed_token_ids: jax.Array,
    sample_tokens: int,
    rng: jax.Array,
) -> str:
    if sample_tokens <= 0:
        return ""

    rng, seed_rng = jax.random.split(rng)
    seed_start = int(
        jax.random.randint(
            seed_rng,
            shape=(),
            minval=0,
            maxval=seed_token_ids.shape[0] - CONTEXT_LENGTH,
        )
    )
    context = seed_token_ids[seed_start : seed_start + CONTEXT_LENGTH]
    generated_token_ids = context[:sample_tokens].tolist()

    for _ in range(max(sample_tokens - len(generated_token_ids), 0)):
        logits = model(context[None, :])
        rng, token_rng = jax.random.split(rng)
        next_token_id = int(jax.random.categorical(token_rng, logits[0, -1]))
        generated_token_ids.append(next_token_id)
        context = jnp.concatenate((context[1:], jnp.asarray([next_token_id], dtype=jnp.int32)))

    return tokenizer.decode_for_display(generated_token_ids)


def main():
    timer = Timer()
    timer.start("total")
    rngs = nnx.Rngs(SEED)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    token_metadata = load_token_shard_metadata(TOKEN_SHARD_ROOT)
    train_shard_path, train_tokens = load_experiment_split(
        TOKEN_SHARD_ROOT,
        "train",
        TRAIN_SHARD_INDEX,
    )
    validation_shard_path, validation_tokens = load_experiment_split(
        TOKEN_SHARD_ROOT,
        "validation",
        VALIDATION_SHARD_INDEX,
    )

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

    rng, validation_rng = jax.random.split(jax.random.key(SEED))
    validation_start_positions = sample_evaluation_positions(
        validation_tokens,
        context_length=CONTEXT_LENGTH,
        subset_size=VALIDATION_SUBSET_EXAMPLES,
        rng=validation_rng,
    )
    loss_tracker = LossTracker()

    for chunk_index, _ in enumerate(range(0, TRAIN_STEPS, TRAIN_CHUNK_LENGTH), start=1):
        train_loss, rng = train_chunk(model, optimizer, train_tokens, rng)
        validation_subset_loss = evaluate_positions(
            validation_tokens,
            validation_start_positions,
            model,
            evaluate_batch_loss,
            CONTEXT_LENGTH,
            EVAL_BATCH_SIZE,
        )

        current_step = chunk_index * TRAIN_CHUNK_LENGTH
        loss_tracker.log(
            step=current_step,
            train_loss=float(train_loss),
            validation_subset_loss=validation_subset_loss,
        )

    train_seconds = timer.stop("train")
    validation_loss = evaluate_split(
        validation_tokens,
        model,
        evaluate_batch_loss,
        CONTEXT_LENGTH,
        EVAL_BATCH_SIZE,
    )
    rng, sample_rng = jax.random.split(rng)
    sample = generate_text(model, tokenizer, train_tokens, SAMPLE_TOKENS, sample_rng)
    loss_history_csv, loss_curve_svg = loss_tracker.save(script_path=Path(__file__))
    sample_path = loss_history_csv.parent / "sample.txt"
    sample_path.write_text(sample + "\n", encoding="utf-8")
    total_seconds = timer.stop("total")

    print(f"token_shard_root={TOKEN_SHARD_ROOT}")
    print(f"tokenizer_path={TOKENIZER_PATH}")
    print(f"token_dtype={token_metadata['token_dtype']}")
    print(f"metadata_shard_tokens={token_metadata['shard_tokens']}")
    print(f"train_shard_path={train_shard_path}")
    print(f"validation_shard_path={validation_shard_path}")
    print(f"train_shard_index={TRAIN_SHARD_INDEX}")
    print(f"validation_shard_index={VALIDATION_SHARD_INDEX}")
    print(f"loaded_train_tokens={train_tokens.shape[0]}")
    print(f"loaded_validation_tokens={validation_tokens.shape[0]}")
    print(f"final_train_loss={loss_tracker.train_losses[-1]:.6f}")
    print(f"validation_loss={validation_loss:.6f}")
    print(f"loss_history_csv={loss_history_csv}")
    print(f"loss_curve_svg={loss_curve_svg}")
    print(f"sample_path={sample_path}")
    print(f"train_seconds={train_seconds:.3f}")
    print(f"steps_per_second={TRAIN_STEPS / train_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
