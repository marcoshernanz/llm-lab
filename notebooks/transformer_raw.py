# %%
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
import jax.nn as jnn

# %%

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "tinyshakespeare.txt"
SEED = 1337
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
CONTEXT_WINDOW = 512
ATTENTION_DIM = 32
LEARNING_RATE = 0.05
TRAIN_STEPS = 5_000
LAYER_NORM_EPS = 1e-5

Array: TypeAlias = jax.Array
Params: TypeAlias = dict[str, Array]

# %%

key = jax.random.key(SEED)
corpus = DATA_PATH.read_text(encoding="utf-8")

vocab_chars = sorted(set(corpus))
char_to_id = {char: idx for idx, char in enumerate(vocab_chars)}
vocab_size = len(char_to_id)

token_ids = jnp.array([char_to_id[ch] for ch in corpus], dtype=jnp.int32)
num_tokens = token_ids.shape[0]

# %%

(
    key,
    token_embedding_key,
    position_embedding_key,
    query_weights_key,
    key_weights_key,
    value_weights_key,
    attention_output_weights_key,
    logit_weights_key,
    logit_bias_key,
    ffn_hidden_weights_key,
    ffn_hidden_bias_key,
    ffn_output_weights_key,
    ffn_output_bias_key,
) = jax.random.split(key, 13)

token_embedding_table = jax.random.normal(token_embedding_key, (vocab_size, EMBEDDING_DIM))
position_embedding_table = jax.random.normal(
    position_embedding_key, (CONTEXT_WINDOW, EMBEDDING_DIM)
)
query_weights = jax.random.normal(query_weights_key, (EMBEDDING_DIM, ATTENTION_DIM))
key_weights = jax.random.normal(key_weights_key, (EMBEDDING_DIM, ATTENTION_DIM))
value_weights = jax.random.normal(value_weights_key, (EMBEDDING_DIM, ATTENTION_DIM))
attention_output_weights = jax.random.normal(
    attention_output_weights_key, (ATTENTION_DIM, EMBEDDING_DIM)
)
logit_weights = jax.random.normal(logit_weights_key, (EMBEDDING_DIM, vocab_size))
logit_bias = jax.random.normal(logit_bias_key, (vocab_size,))
attention_norm_scale = jnp.ones((EMBEDDING_DIM,))
attention_norm_shift = jnp.zeros((EMBEDDING_DIM,))
ffn_hidden_weights = jax.random.normal(ffn_hidden_weights_key, (EMBEDDING_DIM, HIDDEN_DIM))
ffn_hidden_bias = jax.random.normal(ffn_hidden_bias_key, (HIDDEN_DIM,))
ffn_output_weights = jax.random.normal(ffn_output_weights_key, (HIDDEN_DIM, EMBEDDING_DIM))
ffn_output_bias = jax.random.normal(ffn_output_bias_key, (EMBEDDING_DIM,))
ffn_norm_scale = jnp.ones((EMBEDDING_DIM,))
ffn_norm_shift = jnp.zeros((EMBEDDING_DIM,))

params: Params = {
    "token_embedding_table": token_embedding_table,
    "position_embedding_table": position_embedding_table,
    "query_weights": query_weights,
    "key_weights": key_weights,
    "value_weights": value_weights,
    "attention_output_weights": attention_output_weights,
    "attention_norm_scale": attention_norm_scale,
    "attention_norm_shift": attention_norm_shift,
    "ffn_hidden_weights": ffn_hidden_weights,
    "ffn_hidden_bias": ffn_hidden_bias,
    "ffn_output_weights": ffn_output_weights,
    "ffn_output_bias": ffn_output_bias,
    "ffn_norm_scale": ffn_norm_scale,
    "ffn_norm_shift": ffn_norm_shift,
    "logit_weights": logit_weights,
    "logit_bias": logit_bias,
}

# %%


def sample_batch(batch_key: Array) -> tuple[Array, Array]:
    start_positions = jax.random.randint(batch_key, (BATCH_SIZE,), 0, num_tokens - CONTEXT_WINDOW)
    input_positions = start_positions[:, None] + jnp.arange(CONTEXT_WINDOW)
    input_ids = token_ids[input_positions]
    target_ids = token_ids[input_positions + 1]
    return input_ids, target_ids


def loss_fn(params: Params, input_ids: Array, target_ids: Array) -> Array:
    positions = jnp.arange(CONTEXT_WINDOW)
    token_embeddings = params["token_embedding_table"][input_ids]
    position_embeddings = params["position_embedding_table"][positions]
    input_embeddings = token_embeddings + position_embeddings

    queries = input_embeddings @ params["query_weights"]
    keys = input_embeddings @ params["key_weights"]
    values = input_embeddings @ params["value_weights"]

    scores = (queries @ keys.mT) / jnp.sqrt(ATTENTION_DIM)
    causal_mask = jnp.triu(jnp.ones((CONTEXT_WINDOW, CONTEXT_WINDOW), dtype=bool), k=1)
    masked_scores = jnp.where(causal_mask, -jnp.inf, scores)
    attention_weights = jnn.softmax(masked_scores, axis=-1)
    mixed_values = attention_weights @ values
    projected_attention_output = mixed_values @ params["attention_output_weights"]
    attention_residual_output = input_embeddings + projected_attention_output

    normalized_attention_residual = (
        attention_residual_output - attention_residual_output.mean(axis=-1, keepdims=True)
    ) / jnp.sqrt(attention_residual_output.var(axis=-1, keepdims=True) + LAYER_NORM_EPS)
    attention_block_output = (
        params["attention_norm_scale"] * normalized_attention_residual
        + params["attention_norm_shift"]
    )

    ffn_hidden = jnp.tanh(
        attention_block_output @ params["ffn_hidden_weights"] + params["ffn_hidden_bias"]
    )
    ffn_output = ffn_hidden @ params["ffn_output_weights"] + params["ffn_output_bias"]
    ffn_residual_output = ffn_output + attention_block_output
    normalized_ffn_residual = (
        ffn_residual_output - ffn_residual_output.mean(axis=-1, keepdims=True)
    ) / jnp.sqrt(ffn_residual_output.var(axis=-1, keepdims=True) + LAYER_NORM_EPS)
    block_output = params["ffn_norm_scale"] * normalized_ffn_residual + params["ffn_norm_shift"]

    logits = block_output @ params["logit_weights"] + params["logit_bias"]
    log_probs = -jnn.log_softmax(logits, axis=-1)
    loss_per_token = jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return loss_per_token.mean()


@jax.jit
def train_step(params: Params, input_ids: Array, target_ids: Array) -> tuple[Params, Array]:
    loss, grads = jax.value_and_grad(loss_fn)(params, input_ids, target_ids)
    updated_params = jax.tree_util.tree_map(
        lambda param, grad: param - LEARNING_RATE * grad,
        params,
        grads,
    )
    return updated_params, loss


# %%

for step in range(TRAIN_STEPS):
    key, batch_key = jax.random.split(key, 2)
    input_ids, target_ids = sample_batch(batch_key)
    params, loss = train_step(params, input_ids, target_ids)

    if step % 100 == 0:
        print(f"step={step} loss={loss.item():.4f}")
