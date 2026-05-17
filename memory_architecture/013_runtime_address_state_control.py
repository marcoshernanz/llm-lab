"""Memory architecture experiment 012: sparse writable memory on multi-query binding."""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

DEVICE = "mps"
SEED = 1337

CHUNK_SIZE = 16
SEQUENCE_LEN = 128
assert SEQUENCE_LEN % CHUNK_SIZE == 0
EMBEDDING_DIM = 64
NUM_HEADS = 4
assert EMBEDDING_DIM % NUM_HEADS == 0
ADDRESS_DIM = 32
NUM_MEMORY_SLOTS = 64
TOP_K_MEMORY_READS = 8
READ_TEMPERATURE = 0.25
WRITE_TEMPERATURE = 0.25
HIDDEN_DIM = 256
NUM_BLOCKS = 4

BATCH_SIZE = 64
LEARNING_RATE = 3e-3
TRAIN_STEPS = 2_000
EVAL_INTERVAL = 200
EVAL_BATCHES = 32

NUM_FACTS = 8
NUM_KEYS = 16
NUM_VALUES = 16
NUM_NOISE_TOKENS = 32
CANDIDATE_GUESS_BASELINE = 1.0 / NUM_FACTS
RANDOM_VALUE_EXACT_BASELINE = 1.0 / NUM_VALUES
RANDOM_VALUE_CANDIDATE_BASELINE = NUM_FACTS / NUM_VALUES

PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
STORE_TOKEN = "[STORE]"
QUERY_TOKEN = "[QUERY]"
ANSWER_TOKEN = "[ANSWER]"


class CausalSelfAttention(nn.Module):
    """Apply masked multi-head self-attention inside one chunk."""

    def __init__(self):
        """Create the query, key, value, and output projections."""
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_dim = EMBEDDING_DIM // NUM_HEADS
        self.q_proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.k_proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.v_proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.out_proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape embeddings into separate attention heads."""
        batch_size, chunk_size, _ = x.shape
        return x.reshape(batch_size, chunk_size, self.num_heads, self.head_dim).swapaxes(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads back into one embedding axis."""
        batch_size, _, chunk_size, _ = x.shape
        return x.swapaxes(1, 2).reshape(batch_size, chunk_size, self.num_heads * self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention outputs for one batch of chunk embeddings."""
        chunk_size = x.shape[1]

        queries = self.split_heads(self.q_proj(x))
        keys = self.split_heads(self.k_proj(x))
        values = self.split_heads(self.v_proj(x))

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(causal_mask, -torch.inf)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = self.merge_heads(attention_weights @ values)
        return self.out_proj(attended_values)


class SparseLatentAddressRead(nn.Module):
    """Read runtime memory values from each token's nearest address slots."""

    def __init__(self):
        """Create the token-to-address query and output projections."""
        super().__init__()
        self.q_proj = nn.Linear(EMBEDDING_DIM, ADDRESS_DIM, bias=False)
        self.out_proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory_addresses: torch.Tensor,
        memory_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return sparse top-k memory read residuals for one chunk."""
        queries = F.normalize(self.q_proj(x), dim=-1)
        addresses = F.normalize(memory_addresses, dim=-1)

        read_scores = queries @ addresses.T
        read_scores = read_scores / READ_TEMPERATURE
        top_scores, top_indices = torch.topk(
            read_scores,
            k=TOP_K_MEMORY_READS,
            dim=-1,
        )
        read_weights = F.softmax(top_scores, dim=-1)

        batch_size, _, _ = x.shape
        batch_indices = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1)
        selected_memory_values = memory_values[batch_indices, top_indices]
        read_values = (read_weights[..., None] * selected_memory_values).sum(dim=2)
        return self.out_proj(read_values)


class FixedAddressMemoryWrite(nn.Module):
    """Write chunk states into per-example memory values at fixed addresses."""

    def __init__(self):
        """Create projections for write addresses, values, and gates."""
        super().__init__()
        self.q_proj = nn.Linear(EMBEDDING_DIM, ADDRESS_DIM, bias=False)
        self.value_proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
        self.gate_proj = nn.Linear(EMBEDDING_DIM, 1)

    def forward(
        self,
        x: torch.Tensor,
        memory_addresses: torch.Tensor,
        memory_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return memory values after an interpolated token-level write."""
        write_queries = F.normalize(self.q_proj(x), dim=-1)
        addresses = F.normalize(memory_addresses, dim=-1)

        write_scores = write_queries @ addresses.T
        write_scores = write_scores / WRITE_TEMPERATURE
        write_weights = F.softmax(write_scores, dim=-1)
        write_gates = torch.sigmoid(self.gate_proj(x))
        write_weights = write_weights * write_gates

        value_updates = self.value_proj(x)
        slot_update_weight = write_weights.sum(dim=1)
        slot_update_sum = write_weights.transpose(1, 2) @ value_updates
        slot_updates = slot_update_sum / slot_update_weight.clamp_min(1e-6)[..., None]
        slot_gates = 1.0 - torch.exp(-slot_update_weight)

        return torch.lerp(memory_values, slot_updates, slot_gates[..., None])


class FeedForward(nn.Module):
    """Project up, apply a nonlinearity, and project back down."""

    def __init__(self):
        """Create the two linear layers of the MLP."""
        super().__init__()
        self.hidden = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, EMBEDDING_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feed-forward block output."""
        x = F.gelu(self.hidden(x))
        return self.out(x)


class RMSNorm(nn.Module):
    """Scale activations by their root-mean-square magnitude."""

    def __init__(self):
        """Create the learned scale parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(EMBEDDING_DIM))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one embedding vector and apply the learned scale."""
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class DecoderBlock(nn.Module):
    """Apply local attention, memory reading, and one MLP block."""

    def __init__(self):
        """Create the norms, attention, memory, and feed-forward sublayers."""
        super().__init__()
        self.attention_norm = RMSNorm()
        self.attention = CausalSelfAttention()
        self.memory_read_norm = RMSNorm()
        self.memory_read = SparseLatentAddressRead()
        self.feed_forward_norm = RMSNorm()
        self.feed_forward = FeedForward()

    def forward(
        self,
        x: torch.Tensor,
        memory_addresses: torch.Tensor,
        memory_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return the residual output of one decoder block."""
        x = x + self.attention(self.attention_norm(x))
        x = x + self.memory_read(self.memory_read_norm(x), memory_addresses, memory_values)
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Decoder(nn.Module):
    """Stack decoder blocks and update runtime memory after each chunk."""

    def __init__(self) -> None:
        """Create the block stack, fixed addresses, and write module."""
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(NUM_BLOCKS)])
        self.out_norm = RMSNorm()
        self.memory_addresses = nn.Parameter(torch.randn(NUM_MEMORY_SLOTS, ADDRESS_DIM))
        self.memory_write_norm = RMSNorm()
        self.memory_write = FixedAddressMemoryWrite()

    def forward(
        self,
        x: torch.Tensor,
        memory_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one chunk and return updated runtime memory values."""
        for block in self.blocks:
            x = block(x, self.memory_addresses, memory_values)

        x = self.out_norm(x)
        memory_values = self.memory_write(
            self.memory_write_norm(x), self.memory_addresses, memory_values
        )
        return x, memory_values


class LanguageModel(nn.Module):
    """Embed tokens, process chunks with writable memory, and predict logits."""

    def __init__(self, vocab_size: int):
        """Create the embeddings and decoder."""
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_embedding = nn.Embedding(SEQUENCE_LEN, EMBEDDING_DIM)
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for one batch of token ids."""
        batch_size, sequence_len = x.shape
        num_chunks = sequence_len // CHUNK_SIZE
        positions = torch.arange(sequence_len, device=x.device)
        position_embeddings = self.position_embedding(positions).reshape(
            1, num_chunks, CHUNK_SIZE, EMBEDDING_DIM
        )

        x = x.reshape(batch_size, num_chunks, CHUNK_SIZE)
        x = self.token_embedding(x) + position_embeddings

        memory_values = x.new_zeros(batch_size, NUM_MEMORY_SLOTS, EMBEDDING_DIM)
        chunk_outputs: list[torch.Tensor] = []
        for chunk_index in range(num_chunks):
            chunk_output, memory_values = self.decoder(x[:, chunk_index], memory_values)
            chunk_outputs.append(chunk_output)

        x = torch.stack(chunk_outputs, dim=1)
        x = x @ self.token_embedding.weight.T
        x = x.reshape(batch_size, sequence_len, self.vocab_size)
        return x


def build_vocab() -> tuple[list[str], dict[str, int]]:
    """Build the small synthetic vocabulary for delayed recall."""
    vocab_tokens = [
        PAD_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        STORE_TOKEN,
        QUERY_TOKEN,
        ANSWER_TOKEN,
    ]
    vocab_tokens.extend([f"K{idx}" for idx in range(NUM_KEYS)])
    vocab_tokens.extend([f"V{idx}" for idx in range(NUM_VALUES)])
    vocab_tokens.extend([f"N{idx}" for idx in range(NUM_NOISE_TOKENS)])
    token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
    return vocab_tokens, token_to_id


def build_example(token_to_id: dict[str, int]) -> tuple[list[int], list[int], list[int]]:
    """Create one delayed key-value recall sequence and answer positions."""
    fact_tokens = [token_to_id[BOS_TOKEN]]
    key_tokens = [token_to_id[f"K{idx}"] for idx in range(NUM_KEYS)]
    value_tokens = [token_to_id[f"V{idx}"] for idx in range(NUM_VALUES)]
    noise_tokens = [token_to_id[f"N{idx}"] for idx in range(NUM_NOISE_TOKENS)]

    chosen_keys = random.sample(key_tokens, NUM_FACTS)
    chosen_values = random.sample(value_tokens, NUM_FACTS)
    key_value_pairs = list(zip(chosen_keys, chosen_values))

    for key_id, value_id in key_value_pairs:
        fact_tokens.extend([token_to_id[STORE_TOKEN], key_id, value_id])

    query_pairs = key_value_pairs[:]
    random.shuffle(query_pairs)
    suffix_len = len(query_pairs) * 4 + 1

    full_sequence_len = SEQUENCE_LEN + 1
    filler_len = full_sequence_len - len(fact_tokens) - suffix_len
    if filler_len < 0:
        raise ValueError("SEQUENCE_LEN is too small for the delayed-recall layout.")

    filler_tokens = random.choices(noise_tokens, k=filler_len)
    token_ids = fact_tokens + filler_tokens
    answer_positions = []
    for query_key_id, query_value_id in query_pairs:
        token_ids.extend([token_to_id[QUERY_TOKEN], query_key_id, token_to_id[ANSWER_TOKEN]])
        answer_positions.append(len(token_ids) - 1)
        token_ids.append(query_value_id)
    token_ids.append(token_to_id[EOS_TOKEN])
    return token_ids, answer_positions, chosen_values


def sample_batch(
    token_to_id: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample delayed-recall input, target, answer-mask, and candidate tensors."""
    inputs = torch.empty((BATCH_SIZE, SEQUENCE_LEN), dtype=torch.long, device=DEVICE)
    targets = torch.empty((BATCH_SIZE, SEQUENCE_LEN), dtype=torch.long, device=DEVICE)
    answer_mask = torch.zeros((BATCH_SIZE, SEQUENCE_LEN), dtype=torch.float32, device=DEVICE)
    candidate_values = torch.empty((BATCH_SIZE, NUM_FACTS), dtype=torch.long, device=DEVICE)

    for batch_index in range(BATCH_SIZE):
        token_ids, answer_positions, example_candidate_values = build_example(token_to_id)
        inputs[batch_index] = torch.tensor(token_ids[:-1], dtype=torch.long, device=DEVICE)
        targets[batch_index] = torch.tensor(token_ids[1:], dtype=torch.long, device=DEVICE)
        answer_mask[batch_index, answer_positions] = 1.0
        candidate_values[batch_index] = torch.tensor(
            example_candidate_values,
            dtype=torch.long,
            device=DEVICE,
        )

    return inputs, targets, answer_mask, candidate_values


def answer_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    answer_mask: torch.Tensor,
    candidate_values: torch.Tensor,
) -> tuple[float, float]:
    """Return exact answer accuracy and candidate-value accuracy."""
    predictions = logits.argmax(dim=-1)
    is_answer = answer_mask.bool()
    predicted_answers = predictions[is_answer]
    target_answers = targets[is_answer]
    exact_accuracy = (predicted_answers == target_answers).float().mean()
    candidate_hits = (predictions[:, :, None] == candidate_values[:, None, :]).any(dim=-1)
    candidate_accuracy = candidate_hits[is_answer].float().mean()
    return float(exact_accuracy.item()), float(candidate_accuracy.item())


@torch.no_grad()
def estimate_metrics(
    model: LanguageModel,
    token_to_id: dict[str, int],
) -> tuple[float, float, float]:
    """Estimate answer loss, exact accuracy, and candidate accuracy."""
    losses: list[float] = []
    exact_accuracies: list[float] = []
    candidate_accuracies: list[float] = []
    model.eval()

    for _ in range(EVAL_BATCHES):
        inputs, targets, answer_mask, candidate_values = sample_batch(token_to_id)
        logits = model(inputs)
        loss = loss_fn(logits, targets, answer_mask)
        exact_accuracy, candidate_accuracy = answer_metrics(
            logits,
            targets,
            answer_mask,
            candidate_values,
        )
        losses.append(float(loss.item()))
        exact_accuracies.append(exact_accuracy)
        candidate_accuracies.append(candidate_accuracy)

    model.train()
    return (
        sum(losses) / len(losses),
        sum(exact_accuracies) / len(exact_accuracies),
        sum(candidate_accuracies) / len(candidate_accuracies),
    )


def loss_fn(logits: torch.Tensor, targets: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross-entropy only at the answer position."""
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return (per_token_loss * answer_mask).sum() / answer_mask.sum()


def main() -> None:
    """Train the sparse writable-memory model on multi-query binding recall."""
    random.seed(SEED)
    torch.manual_seed(SEED)

    vocab_tokens, token_to_id = build_vocab()
    model = LanguageModel(len(vocab_tokens)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(
        f"candidate_guess_exact_baseline={CANDIDATE_GUESS_BASELINE:.4f} "
        f"random_value_exact_baseline={RANDOM_VALUE_EXACT_BASELINE:.4f} "
        f"random_value_candidate_baseline={RANDOM_VALUE_CANDIDATE_BASELINE:.4f}"
    )

    for step in range(TRAIN_STEPS):
        inputs, targets, answer_mask, _ = sample_batch(token_to_id)
        logits = model(inputs)
        loss = loss_fn(logits, targets, answer_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_log = step == 0 or (step + 1) % EVAL_INTERVAL == 0 or step == TRAIN_STEPS - 1
        if should_log:
            answer_loss, exact_accuracy, candidate_accuracy = estimate_metrics(model, token_to_id)
            print(
                f"step={step + 1} "
                f"batch_answer_loss={loss.item():.4f} "
                f"eval_answer_loss={answer_loss:.4f} "
                f"eval_exact_answer_accuracy={exact_accuracy:.4f} "
                f"eval_candidate_value_accuracy={candidate_accuracy:.4f}"
            )


if __name__ == "__main__":
    main()
