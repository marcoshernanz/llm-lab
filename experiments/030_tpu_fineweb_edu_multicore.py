"""Train the milestone-030 `pmap` multi-core baseline with run metadata."""

import argparse
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

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
from lib.plotting import LossTracker
from lib.run_artifacts import build_run_metadata, print_run_summary, save_run_artifacts
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
    """Keep the milestone-030 multi-core settings explicit and inspectable."""

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
    global_batch_size: int = 128
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
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
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


def parse_args() -> ExperimentConfig:
    """Parse only the few runtime overrides that are likely to matter in practice."""
    parser = argparse.ArgumentParser(
        description="Train one milestone-030 pmap-based TPU multicore point."
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
        "--global-batch-size",
        type=int,
        default=ExperimentConfig.global_batch_size,
        help="Global training batch size before it is split across devices.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=ExperimentConfig.learning_rate,
        help="Optax AdamW learning rate.",
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
        "--weight-decay",
        type=float,
        default=ExperimentConfig.weight_decay,
        help="Optax AdamW weight decay.",
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
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
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


def create_model_graph_and_params(config: ExperimentConfig, vocab_size: int) -> tuple[object, object]:
    """Build one transformer and split it into a static graph and mutable parameter state."""
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_decoder_blocks=config.num_decoder_blocks,
        rngs=nnx.Rngs(config.seed),
    )
    return nnx.split(model, nnx.Param)


def merge_model(graphdef: object, params: object) -> DecoderOnlyTransformer:
    """Rebuild a callable transformer module from graph and parameter state."""
    return nnx.merge(graphdef, params)


def loss_from_params(
    graphdef: object,
    params: object,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """Compute mean next-token cross-entropy from a pure parameter tree."""
    model = merge_model(graphdef, params)
    logits = model(input_ids)
    loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
    return loss_per_token.mean()


def reshape_for_pmap(batch: jax.Array, num_devices: int) -> jax.Array:
    """Reshape a global batch into `(num_devices, per_device_batch, ...)` for `pmap`."""
    if batch.shape[0] % num_devices != 0:
        raise ValueError(
            "Batch size must be divisible by the number of devices for pmap execution. "
            f"Got batch_size={batch.shape[0]} and num_devices={num_devices}."
        )
    per_device_batch_size = batch.shape[0] // num_devices
    return batch.reshape((num_devices, per_device_batch_size, *batch.shape[1:]))


def build_step_functions(graphdef: object) -> tuple[callable, callable, callable]:
    """Build the `pmap` train and eval helpers from a static graph definition."""

    @partial(jax.pmap, axis_name="data", in_axes=(None, 0, 0))
    def loss_and_grads(
        params: object,
        input_ids: jax.Array,
        target_ids: jax.Array,
    ) -> tuple[jax.Array, object]:
        """Compute local loss and average both loss and grads across devices."""

        def local_loss_fn(current_params: object) -> jax.Array:
            return loss_from_params(graphdef, current_params, input_ids, target_ids)

        loss, grads = jax.value_and_grad(local_loss_fn)(params)
        loss = jax.lax.pmean(loss, "data")
        grads = jax.lax.pmean(grads, "data")
        return loss, grads

    @partial(jax.pmap, axis_name="data", in_axes=(None, 0, 0))
    def evaluate_batch_loss_pmapped(
        params: object,
        input_ids: jax.Array,
        target_ids: jax.Array,
    ) -> jax.Array:
        """Evaluate one evenly sharded batch and return one replicated scalar loss per device."""
        loss = loss_from_params(graphdef, params, input_ids, target_ids)
        return jax.lax.pmean(loss, "data")

    @jax.jit
    def evaluate_batch_loss_plain(
        params: object,
        input_ids: jax.Array,
        target_ids: jax.Array,
    ) -> jax.Array:
        """Evaluate one plain unsharded batch."""
        return loss_from_params(graphdef, params, input_ids, target_ids)

    return loss_and_grads, evaluate_batch_loss_plain, evaluate_batch_loss_pmapped


def build_multicore_evaluate_batch_loss(
    evaluate_batch_loss_plain: callable,
    evaluate_batch_loss_pmapped: callable,
    *,
    num_devices: int,
) -> callable:
    """Build one eval helper that chooses pmapped or plain execution per batch."""

    def evaluate_batch_loss_multicore(
        params: object,
        input_ids: jax.Array,
        target_ids: jax.Array,
    ) -> jax.Array:
        """Evaluate one batch, sharding only when the batch divides evenly."""
        if input_ids.shape[0] % num_devices == 0 and input_ids.shape[0] > 0:
            sharded_input_ids = reshape_for_pmap(input_ids, num_devices)
            sharded_target_ids = reshape_for_pmap(target_ids, num_devices)
            return evaluate_batch_loss_pmapped(params, sharded_input_ids, sharded_target_ids)[0]
        return evaluate_batch_loss_plain(params, input_ids, target_ids)

    return evaluate_batch_loss_multicore


def train_chunk(
    graphdef: object,
    params: object,
    optimizer_tx: optax.GradientTransformation,
    opt_state: object,
    tokens: jax.Array,
    config: ExperimentConfig,
    rng: jax.Array,
    loss_and_grads: callable,
    *,
    num_devices: int,
) -> tuple[object, object, jax.Array, jax.Array]:
    """Average several pmapped training batches into one logged chunk."""
    total_loss = jnp.array(0.0, dtype=jnp.float32)

    for _ in range(config.train_chunk_length):
        rng, batch_rng = jax.random.split(rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(config.global_batch_size,),
            minval=0,
            maxval=tokens.shape[0] - config.context_length,
        )
        input_ids, target_ids = build_examples(tokens, start_positions, config.context_length)
        sharded_input_ids = reshape_for_pmap(input_ids, num_devices)
        sharded_target_ids = reshape_for_pmap(target_ids, num_devices)
        loss, grads = loss_and_grads(params, sharded_input_ids, sharded_target_ids)
        mean_loss = loss[0]
        mean_grads = jax.tree.map(lambda value: value[0], grads)
        updates, opt_state = optimizer_tx.update(mean_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        total_loss = total_loss + mean_loss

    return params, opt_state, total_loss / config.train_chunk_length, rng


def generate_text(
    graphdef: object,
    params: object,
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
    model = merge_model(graphdef, params)

    for _ in range(sample_tokens):
        logits = model(context[None, :])
        rng, token_rng = jax.random.split(rng)
        next_token_id = int(jax.random.categorical(token_rng, logits[0, -1]))
        generated_token_ids.append(next_token_id)
        context = jnp.concatenate((context[1:], jnp.asarray([next_token_id], dtype=jnp.int32)))

    return tokenizer.decode_for_display(generated_token_ids)


def main() -> None:
    """Run one milestone-030 `pmap` TPU multicore point end to end."""
    config = parse_args()
    print("status=parsed_config")
    num_devices = jax.local_device_count()
    print(f"status=resolved_devices jax_local_device_count={num_devices}")
    if config.global_batch_size % num_devices != 0:
        raise ValueError(
            "global_batch_size must be divisible by the visible local JAX device count. "
            f"Got global_batch_size={config.global_batch_size} and num_devices={num_devices}."
        )

    per_device_batch_size = config.global_batch_size // num_devices

    timer = Timer()
    timer.start("total")
    tokenizer = load_tokenizer(config.tokenizer_path)
    print("status=loaded_tokenizer")
    train_shard_paths = select_train_shards(config.token_shard_root, config.max_train_shards)
    print(f"status=selected_train_shards count={len(train_shard_paths)}")
    validation_tokens = load_experiment_split(
        config.token_shard_root,
        "validation",
        config.validation_shard_index,
        mmap=config.shard_mmap,
    )
    print(f"status=loaded_validation_tokens count={int(validation_tokens.shape[0])}")
    train_subset_tokens = load_experiment_split(
        config.token_shard_root,
        "train",
        config.train_subset_shard_index,
        mmap=config.shard_mmap,
    )
    print(f"status=loaded_train_subset_tokens count={int(train_subset_tokens.shape[0])}")

    graphdef, params = create_model_graph_and_params(config, tokenizer.vocab_size)
    print("status=created_model_graph_and_params")
    optimizer_tx = optax.adamw(
        learning_rate=config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    opt_state = optimizer_tx.init(params)
    print("status=initialized_optimizer_state")
    loss_and_grads, evaluate_batch_loss_plain, evaluate_batch_loss_pmapped = build_step_functions(
        graphdef
    )
    print("status=built_pmap_step_functions")
    evaluate_batch_loss_multicore = build_multicore_evaluate_batch_loss(
        evaluate_batch_loss_plain,
        evaluate_batch_loss_pmapped,
        num_devices=num_devices,
    )
    print("status=built_multicore_eval_helper")

    rng, validation_rng, train_subset_rng = jax.random.split(jax.random.key(config.seed), 3)
    validation_start_positions = sample_evaluation_positions(
        validation_tokens,
        context_length=config.context_length,
        subset_size=config.validation_subset_examples,
        rng=validation_rng,
    )
    print(f"status=sampled_validation_positions count={int(validation_start_positions.shape[0])}")
    train_subset_start_positions = sample_evaluation_positions(
        train_subset_tokens,
        context_length=config.context_length,
        subset_size=config.validation_subset_examples,
        rng=train_subset_rng,
    )
    print(f"status=sampled_train_subset_positions count={int(train_subset_start_positions.shape[0])}")

    timer.start("train")
    loss_tracker = LossTracker()
    train_tokens = None

    for chunk_index, _ in enumerate(range(0, config.train_steps, config.train_chunk_length)):
        active_train_shard_index = chunk_index % len(train_shard_paths)
        train_tokens = load_token_shard(
            train_shard_paths[active_train_shard_index],
            mmap=config.shard_mmap,
        )
        if chunk_index == 0:
            print(
                "status=starting_first_train_chunk "
                f"train_shard_index={active_train_shard_index} "
                f"loaded_train_tokens={int(train_tokens.shape[0])}"
            )
        params, opt_state, train_loss, rng = train_chunk(
            graphdef,
            params,
            optimizer_tx,
            opt_state,
            train_tokens,
            config,
            rng,
            loss_and_grads,
            num_devices=num_devices,
        )
        if chunk_index == 0:
            print("status=finished_first_train_chunk")
        train_subset_loss = evaluate_positions(
            train_subset_tokens,
            train_subset_start_positions,
            params,
            evaluate_batch_loss_multicore,
            config.context_length,
            config.eval_batch_size,
        )
        if chunk_index == 0:
            print("status=finished_first_train_subset_eval")
        validation_subset_loss = evaluate_positions(
            validation_tokens,
            validation_start_positions,
            params,
            evaluate_batch_loss_multicore,
            config.context_length,
            config.eval_batch_size,
        )
        if chunk_index == 0:
            print("status=finished_first_validation_subset_eval")

        current_step = (chunk_index + 1) * config.train_chunk_length
        loss_tracker.log(
            step=current_step,
            train_loss=float(train_loss),
            train_subset_loss=train_subset_loss,
            validation_subset_loss=validation_subset_loss,
        )

    train_seconds = timer.stop("train")
    if train_tokens is None:
        raise ValueError("No train shard was loaded during training.")

    sample_text = generate_text(
        graphdef,
        params,
        tokenizer,
        validation_tokens,
        config.sample_tokens,
        config.context_length,
        rng,
    )
    sample_model = merge_model(graphdef, params)
    sample_logits = sample_model(validation_tokens[: config.context_length][None, :])
    total_seconds = timer.stop("total")

    run_metadata = build_run_metadata(
        script_path=Path(__file__),
        config=asdict(config),
        execution_target=config.execution_target,
        run_details={
            "batch_size": config.global_batch_size,
            "global_batch_size": config.global_batch_size,
            "per_device_batch_size": per_device_batch_size,
            "sharding_mode": "pmap",
            "parameter_sharding": "replicated parameter tree broadcast to pmap replicas",
            "train_batch_sharding": (
                f"reshaped to ({num_devices}, {per_device_batch_size}, {config.context_length})"
            ),
            "logits_sharding": str(sample_logits.sharding),
            "train_shards_used": len(train_shard_paths),
            "loaded_train_tokens": int(train_tokens.shape[0]),
            "loaded_train_subset_tokens": int(train_subset_tokens.shape[0]),
            "loaded_validation_tokens": int(validation_tokens.shape[0]),
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
        sample_text=sample_text,
        metadata=run_metadata,
        artifacts_root=config.artifacts_root,
    )
    print_run_summary(
        metadata=run_metadata,
        artifacts=artifacts,
        sample_text=sample_text,
    )


if __name__ == "__main__":
    main()
