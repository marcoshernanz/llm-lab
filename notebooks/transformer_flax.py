from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import optax  # pyright: ignore

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
BATCH_SIZE = 64
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_DECODER_BLOCKS = 4
HIDDEN_DIM = 256
CONTEXT_WINDOW = 512
LEARNING_RATE = 1e-3
TRAIN_STEPS = 5_000


class DecoderBlock(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=NUM_HEADS,
            in_features=EMBEDDING_DIM,
            qkv_features=EMBEDDING_DIM,
            out_features=EMBEDDING_DIM,
            use_bias=False,
            rngs=rngs,
        )
        self.attention_norm = nnx.LayerNorm(EMBEDDING_DIM, rngs=rngs)
        self.ffn_in = nnx.Linear(EMBEDDING_DIM, HIDDEN_DIM, rngs=rngs)
        self.ffn_out = nnx.Linear(HIDDEN_DIM, EMBEDDING_DIM, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(EMBEDDING_DIM, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        x = self.attention_norm(x + self.attention(x, mask=mask))
        return self.ffn_norm(x + self.ffn_out(nnx.gelu(self.ffn_in(x))))


class LanguageModel(nnx.Module):
    def __init__(self, vocab_size: int, *, rngs: nnx.Rngs):
        self.token_embedding = nnx.Embed(vocab_size, EMBEDDING_DIM, rngs=rngs)
        self.position_embedding = nnx.Embed(CONTEXT_WINDOW, EMBEDDING_DIM, rngs=rngs)
        self.blocks = nnx.List([DecoderBlock(rngs=rngs) for _ in range(NUM_DECODER_BLOCKS)])
        self.lm_head = nnx.Linear(EMBEDDING_DIM, vocab_size, rngs=rngs)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        positions = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        causal_mask = nnx.make_causal_mask(input_ids, dtype=jnp.bool_)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.lm_head(x)


def loss_fn(model: LanguageModel, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    log_probs = nnx.log_softmax(logits, axis=-1)
    loss_per_token = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@nnx.jit
def train_step(
    model: LanguageModel,
    optimizer: nnx.Optimizer[LanguageModel],
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    loss, grads = nnx.value_and_grad(loss_fn)(model, input_ids, target_ids)
    optimizer.update(model, grads)
    return loss


def sample_batch(batch_key: jax.Array, token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    max_start = token_ids.shape[0] - CONTEXT_WINDOW
    start_positions = jax.random.randint(batch_key, (BATCH_SIZE,), 0, max_start)
    input_positions = start_positions[:, None] + jnp.arange(CONTEXT_WINDOW)
    input_ids = token_ids[input_positions]
    target_ids = token_ids[input_positions + 1]
    return input_ids, target_ids


rngs = nnx.Rngs(SEED)
corpus = DATA_PATH.read_text(encoding="utf-8")
vocab_chars = sorted(set(corpus))
char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
token_ids = jnp.asarray([char_to_id[ch] for ch in corpus], dtype=jnp.int32)

model = LanguageModel(len(vocab_chars), rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE), wrt=nnx.Param)

batch_rng = jax.random.key(SEED)
for step in range(TRAIN_STEPS):
    batch_rng, step_rng = jax.random.split(batch_rng)
    input_ids, target_ids = sample_batch(step_rng, token_ids)
    loss = train_step(model, optimizer, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
