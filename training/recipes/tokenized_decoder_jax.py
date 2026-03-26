from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flax import nnx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax  # pyright: ignore

from models.transformer import LanguageModel
from tokenizer.bpe import BPEModel
from training.config import TrainingConfig


TRAIN_SPLIT_RATIO = 0.8


@dataclass(frozen=True, slots=True)
class DatasetStats:
    vocab_size: int
    train_chars: int
    validation_chars: int
    train_tokens: int
    validation_tokens: int
    train_chars_per_token: float
    validation_chars_per_token: float


@dataclass(slots=True)
class TokenizedDecoderJaxRecipe:
    config: TrainingConfig
    model: LanguageModel
    optimizer: nnx.Optimizer[LanguageModel]
    tokenizer: BPEModel
    train_token_ids: jax.Array
    validation_token_ids: jax.Array
    train_text: str
    validation_text: str
    stats: DatasetStats

    @classmethod
    def create(cls, config: TrainingConfig) -> TokenizedDecoderJaxRecipe:
        text = _load_text(config.data.dataset_path, config.data.text_limit)
        tokenizer = _load_tokenizer(config.data.tokenizer_path)
        train_token_ids, validation_token_ids, train_text, validation_text = _build_token_splits(
            text, tokenizer, train_split_ratio=TRAIN_SPLIT_RATIO
        )
        if (
            train_token_ids.shape[0] <= config.data.context_tokens
            or validation_token_ids.shape[0] <= config.data.context_tokens
        ):
            raise ValueError(
                f"Dataset splits are too small for context length {config.data.context_tokens}. "
                "Need at least one full context window plus one target token in each split."
            )

        rngs = nnx.Rngs(config.run.seed)
        model = LanguageModel(config, tokenizer.vocab_size, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.sgd(config.optimizer.learning_rate), wrt=nnx.Param)
        stats = DatasetStats(
            vocab_size=tokenizer.vocab_size,
            train_chars=len(train_text),
            validation_chars=len(validation_text),
            train_tokens=int(train_token_ids.shape[0]),
            validation_tokens=int(validation_token_ids.shape[0]),
            train_chars_per_token=len(train_text) / int(train_token_ids.shape[0]),
            validation_chars_per_token=len(validation_text) / int(validation_token_ids.shape[0]),
        )
        return cls(
            config=config,
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            train_token_ids=train_token_ids,
            validation_token_ids=validation_token_ids,
            train_text=train_text,
            validation_text=validation_text,
            stats=stats,
        )

    def train_batch(self, rng: jax.Array) -> tuple[jax.Array, float]:
        rng, batch_rng = jax.random.split(rng)
        start_positions = jax.random.randint(
            batch_rng,
            shape=(self.config.train.batch_size,),
            minval=0,
            maxval=self.train_token_ids.shape[0] - self.config.data.context_tokens,
        )
        input_ids, target_ids = _build_examples(
            self.train_token_ids,
            start_positions,
            self.config.data.context_tokens,
        )
        loss = _train_step(self.model, self.optimizer, input_ids, target_ids)
        return rng, float(loss)

    def evaluate_train_loss(self) -> float:
        return _evaluate_split(
            self.train_token_ids,
            self.model,
            context_tokens=self.config.data.context_tokens,
            eval_batch_size=self.config.train.eval_batch_size,
        )

    def evaluate_validation_loss(self) -> float:
        return _evaluate_split(
            self.validation_token_ids,
            self.model,
            context_tokens=self.config.data.context_tokens,
            eval_batch_size=self.config.train.eval_batch_size,
        )

    def generate_sample(self, rng: jax.Array) -> tuple[jax.Array, str]:
        rng, seed_rng = jax.random.split(rng)
        context_tokens = self.config.data.context_tokens
        seed_start = int(
            jax.random.randint(
                seed_rng,
                shape=(),
                minval=0,
                maxval=self.train_token_ids.shape[0] - context_tokens,
            )
        )
        context = self.train_token_ids[seed_start : seed_start + context_tokens]
        generated_token_ids = context[: self.config.train.sample_tokens].tolist()

        for _ in range(max(self.config.train.sample_tokens - len(generated_token_ids), 0)):
            logits = self.model(context[None, :])
            rng, token_rng = jax.random.split(rng)
            next_token_id = int(jax.random.categorical(token_rng, logits[0, -1]))
            generated_token_ids.append(next_token_id)
            context = jnp.concatenate((context[1:], jnp.asarray([next_token_id], dtype=jnp.int32)))

        return rng, _decode_token_ids_for_sample(self.tokenizer, generated_token_ids)


def _load_text(path: Path, text_limit: int | None) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place tinyshakespeare.txt there before running this recipe."
        )
    text = path.read_text(encoding="utf-8")
    if text_limit is not None:
        text = text[:text_limit]
    if len(text) < 2:
        raise ValueError("Dataset is too small. Need at least 2 characters.")
    return text


def _load_tokenizer(path: Path) -> BPEModel:
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer artifact not found at {path}. Train and freeze the tokenizer first."
        )
    return BPEModel.load(path)


def _encode_text(tokenizer: BPEModel, text: str) -> jax.Array:
    return jnp.asarray(tokenizer.encode(text), dtype=jnp.int32)


def _build_token_splits(
    text: str,
    tokenizer: BPEModel,
    *,
    train_split_ratio: float,
) -> tuple[jax.Array, jax.Array, str, str]:
    split_index = int(len(text) * train_split_ratio)
    train_text = text[:split_index]
    validation_text = text[split_index:]
    train_token_ids = _encode_text(tokenizer, train_text)
    validation_token_ids = _encode_text(tokenizer, validation_text)
    return train_token_ids, validation_token_ids, train_text, validation_text


def _build_examples(
    token_ids: jax.Array,
    start_positions: jax.Array,
    context_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    offsets = jnp.arange(context_tokens, dtype=start_positions.dtype)
    input_ids = token_ids[start_positions[:, None] + offsets]
    target_ids = token_ids[start_positions[:, None] + offsets + 1]
    return input_ids, target_ids


def _loss_fn(model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    log_probs = jnn.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@nnx.jit
def _train_step(
    model: LanguageModel,
    optimizer: nnx.Optimizer[LanguageModel],
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    loss, grads = nnx.value_and_grad(_loss_fn)(model, input_ids, target_ids)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def _evaluate_batch_loss(
    model: LanguageModel,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    return _loss_fn(model, input_ids, target_ids)


def _evaluate_split(
    token_ids: jax.Array,
    model: LanguageModel,
    *,
    context_tokens: int,
    eval_batch_size: int,
) -> float:
    max_start = token_ids.shape[0] - context_tokens
    if max_start <= 0:
        raise ValueError(
            f"Dataset split is too small for context length {context_tokens}. "
            "Need at least one full context window plus one target token."
        )

    total_loss = 0.0
    total_examples = 0

    for batch_start in range(0, max_start, eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, max_start)
        start_positions = jnp.arange(batch_start, batch_end, dtype=jnp.int32)
        input_ids, target_ids = _build_examples(token_ids, start_positions, context_tokens)
        batch_loss = _evaluate_batch_loss(model, input_ids, target_ids)
        batch_size = int(start_positions.shape[0])
        total_loss += float(batch_loss) * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def _decode_token_ids_for_sample(tokenizer: BPEModel, token_ids: list[int]) -> str:
    decoded = b"".join(tokenizer.vocab[token_id] for token_id in token_ids)
    return decoded.decode("utf-8", errors="replace")
