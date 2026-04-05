"""Train the milestone-030 profiling baseline with self-describing artifacts."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

import optax  # pyright: ignore
from flax import nnx

import jax
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
from models.transformer import DecoderOnlyTransformer
from tokenizer.bpe import BPEModel

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TOKEN_SHARD_ROOT = ROOT_DIR / "datasets" / "fineweb_edu" / "sample10bt_bpe_16384"
DEFAULT_TOKENIZER_PATH = (
    ROOT_DIR / "artifacts" / "tokenizers" / "fineweb_edu_sample10bt_bpe_16384.json"
)
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ExperimentConfig:
    """Keep the milestone-030 ecosystem settings explicit and easy to inspect."""

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
    weight_decay: float = 0.01
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
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
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


@dataclass(frozen=True)
class ProfileMetrics:
    """Store the coarse timing breakdown for the milestone-030 baseline."""

    train_compile_seconds: float
    eval_compile_seconds: float
    shard_load_seconds: float
    train_chunk_seconds: float
    train_subset_eval_seconds: float
    validation_subset_eval_seconds: float
    training_phase_seconds: float
    sampling_seconds: float
    train_chunks: int
    train_steps: int
    batch_size: int
    context_length: int
    sample_tokens: int

    def to_metadata(self) -> dict[str, float | int]:
        """Flatten the timing breakdown into JSON-safe run-metadata fields."""
        profile_metrics: dict[str, float | int] = {
            "profile_train_compile_seconds": self.train_compile_seconds,
            "profile_eval_compile_seconds": self.eval_compile_seconds,
            "profile_shard_load_seconds": self.shard_load_seconds,
            "profile_train_chunk_seconds": self.train_chunk_seconds,
            "profile_train_subset_eval_seconds": self.train_subset_eval_seconds,
            "profile_validation_subset_eval_seconds": self.validation_subset_eval_seconds,
            "profile_training_phase_seconds": self.training_phase_seconds,
            "profile_sampling_seconds": self.sampling_seconds,
            "profile_train_chunks": self.train_chunks,
        }
        if self.train_chunks > 0:
            profile_metrics["profile_train_chunk_seconds_mean"] = (
                self.train_chunk_seconds / self.train_chunks
            )
        if self.train_steps > 0:
            profile_metrics["profile_train_step_seconds"] = (
                self.train_chunk_seconds / self.train_steps
            )
        train_tokens_seen = self.train_steps * self.batch_size * self.context_length
        if self.train_chunk_seconds > 0:
            profile_metrics["profile_train_tokens_per_second"] = (
                train_tokens_seen / self.train_chunk_seconds
            )

        training_other_seconds = max(
            0.0,
            self.training_phase_seconds
            - self.shard_load_seconds
            - self.train_chunk_seconds
            - self.train_subset_eval_seconds
            - self.validation_subset_eval_seconds,
        )
        profile_metrics["profile_training_other_seconds"] = training_other_seconds

        if self.sample_tokens > 0 and self.sampling_seconds > 0:
            profile_metrics["profile_sample_tokens_per_second"] = (
                self.sample_tokens / self.sampling_seconds
            )

        return profile_metrics


def parse_args() -> ExperimentConfig:
    """Parse only the few profiling overrides worth changing between runs."""
    parser = argparse.ArgumentParser(
        description="Train one milestone-030 TPU profiling point with run metadata."
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
        "--batch-size",
        type=int,
        default=ExperimentConfig.batch_size,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=ExperimentConfig.learning_rate,
        help="Optax AdamW learning rate.",
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
    args = parser.parse_args()

    config = ExperimentConfig(
        token_shard_root=args.token_shard_root,
        tokenizer_path=args.tokenizer_path,
        artifacts_root=args.artifacts_root,
        execution_target=args.execution_target,
        max_train_shards=args.max_train_shards,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_steps=args.train_steps,
        train_chunk_length=args.train_chunk_length,
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


def create_model_and_optimizer(
    config: ExperimentConfig,
    *,
    vocab_size: int,
) -> tuple[DecoderOnlyTransformer, nnx.Optimizer[DecoderOnlyTransformer]]:
    """Build a fresh model and optimizer pair for one run or warmup pass."""
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_decoder_blocks=config.num_decoder_blocks,
        rngs=nnx.Rngs(config.seed),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=config.learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        ),
        wrt=nnx.Param,
    )
    return model, optimizer


def measure_call(
    function: Callable[[], ResultT],
    *,
    block: bool = False,
) -> tuple[ResultT, float]:
    """Measure one callable and optionally block on pending JAX work."""
    start = perf_counter()
    result = function()
    if block:
        result = jax.block_until_ready(result)
    return result, perf_counter() - start


def sample_training_batch(
    tokens: jax.Array,
    *,
    batch_size: int,
    context_length: int,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample one representative training batch from a token shard."""
    rng, batch_rng = jax.random.split(rng)
    start_positions = jax.random.randint(
        batch_rng,
        shape=(batch_size,),
        minval=0,
        maxval=tokens.shape[0] - context_length,
    )
    input_ids, target_ids = build_examples(tokens, start_positions, context_length)
    return input_ids, target_ids, rng


def build_eval_batch(
    tokens: jax.Array,
    start_positions: jax.Array,
    *,
    context_length: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Build one representative evaluation batch from chosen start positions."""
    batch_positions = start_positions[: min(batch_size, start_positions.shape[0])]
    return build_examples(tokens, batch_positions, context_length)


def loss_fn(
    model: DecoderOnlyTransformer,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """Compute mean next-token cross-entropy for one batch."""
    logits = model(input_ids)
    loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
    return loss_per_token.mean()


@nnx.jit
def train_step(
    model: DecoderOnlyTransformer,
    optimizer: nnx.Optimizer[DecoderOnlyTransformer],
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """Run one Optax AdamW step on a batch of token examples."""
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
    """Run one milestone-030 TPU profiling point end to end."""
    config = parse_args()

    timer = Timer()
    timer.start("total")
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
    warmup_input_ids, warmup_target_ids, _ = sample_training_batch(
        train_subset_tokens,
        batch_size=config.batch_size,
        context_length=config.context_length,
        rng=rng,
    )
    warmup_eval_input_ids, warmup_eval_target_ids = build_eval_batch(
        validation_tokens,
        validation_start_positions,
        context_length=config.context_length,
        batch_size=config.eval_batch_size,
    )
    warmup_model, warmup_optimizer = create_model_and_optimizer(
        config,
        vocab_size=tokenizer.vocab_size,
    )
    _, train_compile_seconds = measure_call(
        lambda: train_step(
            warmup_model,
            warmup_optimizer,
            warmup_input_ids,
            warmup_target_ids,
        ),
        block=True,
    )
    _, eval_compile_seconds = measure_call(
        lambda: evaluate_batch_loss(
            warmup_model,
            warmup_eval_input_ids,
            warmup_eval_target_ids,
        ),
        block=True,
    )
    del warmup_model, warmup_optimizer
    model, optimizer = create_model_and_optimizer(
        config,
        vocab_size=tokenizer.vocab_size,
    )
    loss_tracker = LossTracker()
    train_tokens = None
    shard_load_seconds = 0.0
    train_chunk_seconds = 0.0
    train_subset_eval_seconds = 0.0
    validation_subset_eval_seconds = 0.0

    timer.start("train")
    for chunk_index, _ in enumerate(range(0, config.train_steps, config.train_chunk_length)):
        active_train_shard_index = chunk_index % len(train_shard_paths)
        train_tokens, current_shard_load_seconds = measure_call(
            lambda: load_token_shard(
                train_shard_paths[active_train_shard_index],
                mmap=config.shard_mmap,
            )
        )
        shard_load_seconds += current_shard_load_seconds
        (train_loss, rng), current_train_chunk_seconds = measure_call(
            lambda: train_chunk(model, optimizer, train_tokens, config, rng),
            block=True,
        )
        train_chunk_seconds += current_train_chunk_seconds
        train_subset_loss, current_train_subset_eval_seconds = measure_call(
            lambda: evaluate_positions(
                train_subset_tokens,
                train_subset_start_positions,
                model,
                evaluate_batch_loss,
                config.context_length,
                config.eval_batch_size,
            )
        )
        train_subset_eval_seconds += current_train_subset_eval_seconds
        validation_subset_loss, current_validation_subset_eval_seconds = measure_call(
            lambda: evaluate_positions(
                validation_tokens,
                validation_start_positions,
                model,
                evaluate_batch_loss,
                config.context_length,
                config.eval_batch_size,
            )
        )
        validation_subset_eval_seconds += current_validation_subset_eval_seconds

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
    sample, sampling_seconds = measure_call(
        lambda: generate_text(
            model,
            tokenizer,
            train_tokens,
            config.sample_tokens,
            config.context_length,
            sample_rng,
        )
    )
    total_seconds = timer.stop("total")
    profile_metrics = ProfileMetrics(
        train_compile_seconds=train_compile_seconds,
        eval_compile_seconds=eval_compile_seconds,
        shard_load_seconds=shard_load_seconds,
        train_chunk_seconds=train_chunk_seconds,
        train_subset_eval_seconds=train_subset_eval_seconds,
        validation_subset_eval_seconds=validation_subset_eval_seconds,
        training_phase_seconds=train_seconds,
        sampling_seconds=sampling_seconds,
        train_chunks=config.train_steps // config.train_chunk_length,
        train_steps=config.train_steps,
        batch_size=config.batch_size,
        context_length=config.context_length,
        sample_tokens=config.sample_tokens,
    )

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
            **profile_metrics.to_metadata(),
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
