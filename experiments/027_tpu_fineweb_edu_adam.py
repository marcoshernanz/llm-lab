"""Train the milestone-027 Adam baseline with self-describing artifacts."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flax import nnx

import jax
import jax.nn as jnn
import jax.numpy as jnp

from lib.data import (
    build_examples,
    list_token_shards,
    load_token_shard,
    load_tokenizer,
)
from lib.eval import evaluate_positions, sample_evaluation_positions
from lib.run_artifacts import build_run_metadata, print_run_summary, save_run_artifacts
from lib.plotting import LossTracker
from lib.timer import Timer
from lib.optimizers import apply_adam, init_adam_state
from models.transformer import DecoderOnlyTransformer
from tokenizer.bpe import BPEModel

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TOKEN_SHARD_ROOT = ROOT_DIR / "datasets" / "fineweb_edu" / "sample10bt_bpe_16384"
DEFAULT_TOKENIZER_PATH = (
    ROOT_DIR / "artifacts" / "tokenizers" / "fineweb_edu_sample10bt_bpe_16384.json"
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Keep the milestone-027 Adam settings explicit and easy to inspect."""

    token_shard_root: Path = DEFAULT_TOKEN_SHARD_ROOT
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH
    artifacts_root: Path | None = None
    execution_target: str | None = None
    seed: int = 0
    max_train_shards: int | None = 10
    validation_shard_index: int = 0
    train_subset_shard_index: int = 0
    shard_mmap: bool = True
    eval_batch_size: int = 64
    batch_size: int = 128
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    train_steps: int = 20_000
    train_chunk_length: int = 100
    validation_subset_examples: int = 256
    sample_tokens: int = 60
    embedding_dim: int = 128
    num_heads: int = 8
    num_decoder_blocks: int = 8
    hidden_dim: int = 256
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
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.beta1 < 0 or self.beta1 >= 1:
            raise ValueError("beta1 must be in the half-open interval [0, 1)")
        if self.beta2 < 0 or self.beta2 >= 1:
            raise ValueError("beta2 must be in the half-open interval [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
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
        if self.train_subset_shard_index < 0:
            raise ValueError("train_subset_shard_index must be non-negative")


def parse_args() -> ExperimentConfig:
    """Parse the small set of runtime overrides useful on TPU notebooks."""
    parser = argparse.ArgumentParser(
        description="Train one milestone-027 TPU Adam point with run metadata."
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
        "--artifacts-root",
        type=Path,
        default=None,
        help="Optional output root for experiment artifacts.",
    )
    parser.add_argument(
        "--execution-target",
        type=str,
        default=None,
        help="Optional human-readable runtime label such as 'Kaggle TPU v5e-8'.",
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
        "--batch-size",
        type=int,
        default=ExperimentConfig.batch_size,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=ExperimentConfig.learning_rate,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=ExperimentConfig.beta1,
        help="Adam first-moment decay.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=ExperimentConfig.beta2,
        help="Adam second-moment decay.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=ExperimentConfig.epsilon,
        help="Adam denominator stabilizer.",
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
        artifacts_root=args.artifacts_root,
        execution_target=args.execution_target,
        max_train_shards=args.max_train_shards,
        validation_shard_index=args.validation_shard_index,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
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
    first_moment: nnx.State[Any, Any],
    second_moment: nnx.State[Any, Any],
    step: jax.Array,
    input_ids: jax.Array,
    target_ids: jax.Array,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[jax.Array, nnx.State[Any, Any], nnx.State[Any, Any], jax.Array]:
    """Run one Adam step on a batch of token examples."""
    loss, grads = nnx.value_and_grad(loss_fn)(model, input_ids, target_ids)
    first_moment, second_moment, step = apply_adam(
        model, grads, first_moment, second_moment, step, learning_rate, beta1, beta2, epsilon
    )
    return loss, first_moment, second_moment, step


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
    first_moment: nnx.State[Any, Any],
    second_moment: nnx.State[Any, Any],
    step: jax.Array,
    tokens: jax.Array,
    config: ExperimentConfig,
    rng: jax.Array,
) -> tuple[jax.Array, nnx.State[Any, Any], nnx.State[Any, Any], jax.Array, jax.Array]:
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
        loss, first_moment, second_moment, step = train_step(
            model,
            first_moment,
            second_moment,
            step,
            input_ids,
            target_ids,
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.epsilon,
        )
        total_loss = total_loss + loss
    return total_loss / config.train_chunk_length, first_moment, second_moment, step, rng


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
    """Run one milestone-027 TPU Adam point end to end."""
    config = parse_args()

    timer = Timer()
    timer.start("total")
    rngs = nnx.Rngs(config.seed)
    tokenizer = load_tokenizer(config.tokenizer_path)
    train_shard_paths = select_train_shards(config.token_shard_root, config.max_train_shards)
    validation_tokens = load_experiment_split(
        config.token_shard_root,
        "validation",
        config.validation_shard_index,
        mmap=config.shard_mmap,
    )
    train_subset_tokens = load_experiment_split(
        config.token_shard_root,
        "train",
        config.train_subset_shard_index,
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
    first_moment, second_moment, optimizer_step = init_adam_state(model)
    timer.start("train")

    rng, validation_rng, train_subset_rng = jax.random.split(jax.random.key(config.seed), 3)
    validation_start_positions = sample_evaluation_positions(
        validation_tokens,
        context_length=config.context_length,
        subset_size=config.validation_subset_examples,
        rng=validation_rng,
    )
    train_subset_start_positions = sample_evaluation_positions(
        train_subset_tokens,
        context_length=config.context_length,
        subset_size=config.validation_subset_examples,
        rng=train_subset_rng,
    )
    loss_tracker = LossTracker()
    train_tokens = None

    for chunk_index, _ in enumerate(range(0, config.train_steps, config.train_chunk_length)):
        active_train_shard_index = chunk_index % len(train_shard_paths)
        train_tokens = load_token_shard(
            train_shard_paths[active_train_shard_index],
            mmap=config.shard_mmap,
        )
        train_loss, first_moment, second_moment, optimizer_step, rng = train_chunk(
            model,
            first_moment,
            second_moment,
            optimizer_step,
            train_tokens,
            config,
            rng,
        )
        train_subset_loss = evaluate_positions(
            train_subset_tokens,
            train_subset_start_positions,
            model,
            evaluate_batch_loss,
            config.context_length,
            config.eval_batch_size,
        )
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
            train_subset_loss=train_subset_loss,
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
    total_seconds = timer.stop("total")

    metadata = build_run_metadata(
        script_path=Path(__file__),
        config=asdict(config),
        execution_target=config.execution_target,
        run_details={
            "train_shards_used": len(train_shard_paths),
            "loaded_train_tokens": train_tokens.shape[0],
            "loaded_train_subset_tokens": train_subset_tokens.shape[0],
            "loaded_validation_tokens": validation_tokens.shape[0],
        },
        run_metrics={
            "final_train_loss": loss_tracker.train_losses[-1],
            "final_train_subset_loss": loss_tracker.train_subset_losses[-1],
            "final_validation_subset_loss": loss_tracker.validation_subset_losses[-1],
            "train_seconds": train_seconds,
            "total_seconds": total_seconds,
        },
    )
    artifacts = save_run_artifacts(
        script_path=Path(__file__),
        loss_tracker=loss_tracker,
        sample_text=sample,
        metadata=metadata,
        artifacts_root=config.artifacts_root,
    )
    print_run_summary(
        metadata=metadata,
        artifacts=artifacts,
        sample_text=sample,
    )


if __name__ == "__main__":
    main()
