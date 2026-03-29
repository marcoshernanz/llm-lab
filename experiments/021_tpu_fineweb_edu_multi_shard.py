"""Train the milestone-021 FineWeb multi-shard baseline on a TPU runtime."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import optax  # pyright: ignore
from flax import nnx

import jax
import jax.nn as jnn
import jax.numpy as jnp

from lib.data import (
    build_examples,
    list_token_shards,
    load_token_shard,
    load_token_shard_metadata,
    load_tokenizer,
)
from lib.eval import evaluate_positions, sample_evaluation_positions
from lib.plotting import LossTracker
from lib.timer import Timer
from models.transformer import DecoderOnlyTransformer
from tokenizer.bpe import BPEModel

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TOKEN_SHARD_ROOT = ROOT_DIR / "datasets" / "fineweb_edu" / "sample10bt_bpe_16384"
DEFAULT_TOKENIZER_PATH = (
    ROOT_DIR / "artifacts" / "tokenizers" / "fineweb_edu_sample10bt_bpe_16384.json"
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Keep the milestone-021 settings explicit and easy to inspect."""

    token_shard_root: Path = DEFAULT_TOKEN_SHARD_ROOT
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH
    seed: int = 0
    max_train_shards: int | None = 10
    validation_shard_index: int = 0
    shard_mmap: bool = True
    eval_batch_size: int = 32
    batch_size: int = 8
    learning_rate: float = 0.02
    train_steps: int = 2_000
    train_chunk_length: int = 40
    validation_subset_examples: int = 256
    sample_tokens: int = 60
    embedding_dim: int = 64
    num_heads: int = 4
    num_decoder_blocks: int = 4
    hidden_dim: int = 128
    context_length: int = 64

    def validate(self) -> None:
        """Reject invalid experiment settings early."""
        if self.train_steps <= 0:
            raise ValueError("train_steps must be positive")
        if self.train_chunk_length <= 0:
            raise ValueError("train_chunk_length must be positive")
        if self.train_steps % self.train_chunk_length != 0:
            raise ValueError("train_steps must be divisible by train_chunk_length")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.validation_subset_examples <= 0:
            raise ValueError("validation_subset_examples must be positive")
        if self.sample_tokens < 0:
            raise ValueError("sample_tokens must be non-negative")
        if self.max_train_shards is not None and self.max_train_shards <= 0:
            raise ValueError("max_train_shards must be positive when provided")


def parse_args() -> ExperimentConfig:
    """Parse the small set of runtime overrides useful in Colab."""
    parser = argparse.ArgumentParser(
        description="Train the milestone-021 TPU multi-shard FineWeb baseline."
    )
    parser.add_argument(
        "--token-shard-root",
        type=Path,
        default=DEFAULT_TOKEN_SHARD_ROOT,
        help="Directory containing train_*.npy, validation_*.npy, and metadata.json.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer artifact used for decoding samples.",
    )
    parser.add_argument(
        "--max-train-shards",
        type=int,
        default=ExperimentConfig.max_train_shards,
        help="Maximum number of train shards to rotate across.",
    )
    parser.add_argument(
        "--validation-shard-index",
        type=int,
        default=ExperimentConfig.validation_shard_index,
        help="Validation shard index used for the fixed validation subset.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=ExperimentConfig.train_steps,
        help="Total optimizer steps.",
    )
    parser.add_argument(
        "--train-chunk-length",
        type=int,
        default=ExperimentConfig.train_chunk_length,
        help="Number of optimizer steps averaged into one logged point.",
    )
    parser.add_argument(
        "--no-shard-mmap",
        action="store_true",
        help="Disable mmap_mode when loading token shards with jax.numpy.load.",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        token_shard_root=args.token_shard_root,
        tokenizer_path=args.tokenizer_path,
        max_train_shards=args.max_train_shards,
        validation_shard_index=args.validation_shard_index,
        train_steps=args.train_steps,
        train_chunk_length=args.train_chunk_length,
        shard_mmap=not args.no_shard_mmap,
    )
    config.validate()
    return config


def load_experiment_split(
    root_dir: Path,
    split: str,
    shard_index: int,
    *,
    mmap: bool,
) -> jax.Array:
    """Load one validation shard or a chosen train shard for this experiment."""
    shard_paths = list_token_shards(root_dir, split)
    if shard_index < 0 or shard_index >= len(shard_paths):
        raise ValueError(
            f"{split} shard index {shard_index} is out of range for {root_dir}. "
            f"Available {split} shards: {len(shard_paths)}."
        )
    return load_token_shard(shard_paths[shard_index], mmap=mmap)


def select_train_shards(root_dir: Path, max_train_shards: int | None) -> list[Path]:
    """Select the train shards used for this run."""
    shard_paths = list_token_shards(root_dir, "train")
    if max_train_shards is None:
        return shard_paths
    return shard_paths[:max_train_shards]


def load_train_shard_for_chunk(
    train_shard_paths: list[Path],
    chunk_index: int,
    *,
    mmap: bool,
) -> jax.Array:
    """Rotate through the selected train shards, one shard per logged chunk."""
    active_train_shard_index = chunk_index % len(train_shard_paths)
    return load_token_shard(train_shard_paths[active_train_shard_index], mmap=mmap)


def loss_fn(
    model: DecoderOnlyTransformer,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """Compute mean next-token cross-entropy for one batch."""
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
    """Run one optimizer step on a batch of token examples."""
    loss, grads = nnx.value_and_grad(loss_fn)(model, input_ids, target_ids)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def evaluate_batch_loss(
    model: DecoderOnlyTransformer,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """Evaluate batch loss without updating the model."""
    return loss_fn(model, input_ids, target_ids)


def train_chunk(
    model: DecoderOnlyTransformer,
    optimizer: nnx.Optimizer[DecoderOnlyTransformer],
    tokens: jax.Array,
    config: ExperimentConfig,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Average several random training batches into one logged chunk."""
    total_loss = jnp.array(0.0, dtype=jnp.float32)
    for _ in range(config.train_chunk_length):
        rng, batch_rng = jax.random.split(rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(config.batch_size,),
            minval=0,
            maxval=tokens.shape[0] - config.context_length,
        )
        input_ids, target_ids = build_examples(tokens, start_positions, config.context_length)
        total_loss = total_loss + train_step(model, optimizer, input_ids, target_ids)
    return total_loss / config.train_chunk_length, rng


def generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: BPEModel,
    seed_token_ids: jax.Array,
    sample_tokens: int,
    context_length: int,
    rng: jax.Array,
) -> str:
    """Sample text from the trained model using a random context window."""
    if sample_tokens <= 0:
        return ""

    rng, seed_rng = jax.random.split(rng)
    seed_start = int(
        jax.random.randint(
            seed_rng,
            shape=(),
            minval=0,
            maxval=seed_token_ids.shape[0] - context_length,
        )
    )
    context = seed_token_ids[seed_start : seed_start + context_length]
    generated_token_ids: list[int] = []

    for _ in range(sample_tokens):
        logits = model(context[None, :])
        rng, token_rng = jax.random.split(rng)
        next_token_id = int(jax.random.categorical(token_rng, logits[0, -1]))
        generated_token_ids.append(next_token_id)
        context = jnp.concatenate((context[1:], jnp.asarray([next_token_id], dtype=jnp.int32)))

    return tokenizer.decode_for_display(generated_token_ids)


def main() -> None:
    """Run the milestone-021 TPU multi-shard baseline end to end."""
    config = parse_args()

    timer = Timer()
    timer.start("total")
    rngs = nnx.Rngs(config.seed)
    tokenizer = load_tokenizer(config.tokenizer_path)
    token_metadata = load_token_shard_metadata(config.token_shard_root)
    train_shard_paths = select_train_shards(config.token_shard_root, config.max_train_shards)
    validation_tokens = load_experiment_split(
        config.token_shard_root,
        "validation",
        config.validation_shard_index,
        mmap=config.shard_mmap,
    )

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_decoder_blocks=config.num_decoder_blocks,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.sgd(config.learning_rate), wrt=nnx.Param)
    timer.start("train")

    rng, validation_rng = jax.random.split(jax.random.key(config.seed))
    validation_start_positions = sample_evaluation_positions(
        validation_tokens,
        context_length=config.context_length,
        subset_size=config.validation_subset_examples,
        rng=validation_rng,
    )
    loss_tracker = LossTracker()
    train_tokens = None

    for chunk_index, _ in enumerate(range(0, config.train_steps, config.train_chunk_length)):
        train_tokens = load_train_shard_for_chunk(
            train_shard_paths,
            chunk_index,
            mmap=config.shard_mmap,
        )
        train_loss, rng = train_chunk(model, optimizer, train_tokens, config, rng)
        validation_subset_loss = evaluate_positions(
            validation_tokens,
            validation_start_positions,
            model,
            evaluate_batch_loss,
            config.context_length,
            config.eval_batch_size,
        )

        current_step = (chunk_index + 1) * config.train_chunk_length
        loss_tracker.log(
            step=current_step,
            train_loss=float(train_loss),
            validation_subset_loss=validation_subset_loss,
        )

    train_seconds = timer.stop("train")
    rng, sample_rng = jax.random.split(rng)
    if train_tokens is None:
        raise ValueError("No train shard was loaded during training.")
    sample = generate_text(
        model,
        tokenizer,
        train_tokens,
        config.sample_tokens,
        config.context_length,
        sample_rng,
    )
    loss_history_csv, loss_curve_svg = loss_tracker.save(script_path=Path(__file__))
    sample_path = loss_history_csv.parent / "sample.txt"
    sample_path.write_text(sample + "\n", encoding="utf-8")
    total_seconds = timer.stop("total")

    devices = jax.devices()
    print(f"jax_default_backend={jax.default_backend()}")
    print(f"jax_device_count={len(devices)}")
    print(f"token_shard_root={config.token_shard_root}")
    print(f"tokenizer_path={config.tokenizer_path}")
    print(f"token_dtype={token_metadata['token_dtype']}")
    print(f"train_shards_used={len(train_shard_paths)}")
    print(f"max_train_shards={config.max_train_shards}")
    print(f"validation_shard_index={config.validation_shard_index}")
    print(f"shard_mmap={config.shard_mmap}")
    print(f"loaded_train_tokens={train_tokens.shape[0]}")
    print(f"loaded_validation_tokens={validation_tokens.shape[0]}")
    print(f"final_train_loss={loss_tracker.train_losses[-1]:.6f}")
    print(f"final_validation_subset_loss={loss_tracker.validation_subset_losses[-1]:.6f}")
    print(f"loss_history_csv={loss_history_csv}")
    print(f"loss_curve_svg={loss_curve_svg}")
    print(f"sample_path={sample_path}")
    print(f"train_seconds={train_seconds:.3f}")
    print(f"steps_per_second={config.train_steps / train_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f'sample="""\n{sample}\n"""')


if __name__ == "__main__":
    main()
