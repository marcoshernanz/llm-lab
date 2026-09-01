"""Small controlled probe for DeepSeek-V4-style Anticipatory Routing.

This is not a language-model result.  It is a mechanism and implementation probe that asks whether
cached historical routes change recovery after a deliberately injected MoE expert outlier.  The
synthetic autoregressive task keeps the run cheap enough for a laptop while preserving the relevant
closed loop: an early expert changes the hidden state seen by routers in later blocks.

The compute-matched synchronous control performs the same no-gradient future-batch forward passes
as the historical condition, but discards their cached routes.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


Condition = Literal["synchronous", "historical"]


@dataclass(frozen=True)
class Config:
    steps: int = 220
    shock_step: int = 140
    delay: int = 8
    recovery_steps: int = 64
    batch_size: int = 16
    sequence_length: int = 48
    vocabulary_size: int = 36
    num_domains: int = 4
    model_dim: int = 64
    num_heads: int = 4
    num_blocks: int = 3
    num_experts: int = 4
    active_experts: int = 2
    expert_hidden_dim: int = 96
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    balance_weight: float = 0.01
    shock_scale: float = 5.0

    def validate(self) -> None:
        assert 0 < self.shock_step < self.steps
        assert self.shock_step + self.recovery_steps <= self.steps
        assert 0 < self.delay < self.recovery_steps
        assert self.model_dim % self.num_heads == 0
        assert 1 <= self.active_experts < self.num_experts
        assert self.vocabulary_size >= 32 + self.num_domains


@dataclass
class MoEObservation:
    executed_routes: torch.Tensor
    synchronous_routes: torch.Tensor
    margin: torch.Tensor
    loads: torch.Tensor
    max_expert_output: torch.Tensor


@dataclass
class StepObservation:
    step: int
    loss: float
    task_loss: float
    balance_loss: float
    route_disagreement: float
    route_margin: float
    max_expert_output: float
    first_layer_target_load: float
    gradient_norm: float


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(attended.transpose(1, 2).reshape(batch, length, dim))


class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


class SparseMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.active_experts = config.active_experts
        self.router = nn.Linear(config.model_dim, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(config.model_dim, config.expert_hidden_dim) for _ in range(config.num_experts)
        )

    def forward(
        self, x: torch.Tensor, forced_routes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, MoEObservation]:
        batch, length, dim = x.shape
        flat_x = x.reshape(batch * length, dim)
        logits = self.router(flat_x)
        probabilities = logits.softmax(dim=-1)
        ranked = logits.topk(self.active_experts + 1, dim=-1).indices
        synchronous_routes = ranked[:, : self.active_experts]
        routes = (
            synchronous_routes
            if forced_routes is None
            else forced_routes.reshape_as(synchronous_routes)
        )

        selected_probabilities = probabilities.gather(1, routes)
        selected_probabilities = selected_probabilities / selected_probabilities.sum(
            dim=-1, keepdim=True
        )
        output = torch.zeros_like(flat_x)
        max_expert_output = torch.zeros((), device=x.device)

        for expert_index, expert in enumerate(self.experts):
            token_slots = (routes == expert_index).nonzero(as_tuple=False)
            if token_slots.numel() == 0:
                continue
            token_indices = token_slots[:, 0]
            slot_indices = token_slots[:, 1]
            expert_output = expert(flat_x[token_indices])
            max_expert_output = torch.maximum(max_expert_output, expert_output.detach().abs().max())
            weights = selected_probabilities[token_indices, slot_indices, None]
            output.index_add_(0, token_indices, expert_output * weights)

        assignment_fraction = (
            F.one_hot(routes, num_classes=self.num_experts).float().mean(dim=(0, 1))
        )
        mean_probability = probabilities.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(assignment_fraction.detach() * mean_probability)
        kth_score = logits.gather(1, ranked[:, self.active_experts - 1 : self.active_experts])
        rejected_score = logits.gather(1, ranked[:, self.active_experts : self.active_experts + 1])
        observation = MoEObservation(
            executed_routes=routes.view(batch, length, self.active_experts).detach(),
            synchronous_routes=synchronous_routes.view(batch, length, self.active_experts).detach(),
            margin=(kth_score - rejected_score).detach(),
            loads=assignment_fraction.detach(),
            max_expert_output=max_expert_output,
        )
        return output.view(batch, length, dim), balance_loss, observation


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.model_dim)
        self.attention = CausalSelfAttention(config.model_dim, config.num_heads)
        self.moe_norm = nn.RMSNorm(config.model_dim)
        self.moe = SparseMoE(config)

    def forward(
        self, x: torch.Tensor, forced_routes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, MoEObservation]:
        x = x + self.attention(self.attention_norm(x))
        moe_output, balance_loss, observation = self.moe(self.moe_norm(x), forced_routes)
        return x + moe_output, balance_loss, observation


class TinyMoELanguageModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.model_dim)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_blocks))
        self.final_norm = nn.RMSNorm(config.model_dim)
        self.head = nn.Linear(config.model_dim, config.vocabulary_size, bias=False)
        self.head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, forced_routes: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, list[MoEObservation]]:
        x = self.embedding(tokens)
        balance_loss = torch.zeros((), device=x.device)
        observations: list[MoEObservation] = []
        for block_index, block in enumerate(self.blocks):
            block_routes = None if forced_routes is None else forced_routes[block_index]
            x, block_balance, observation = block(x, block_routes)
            balance_loss = balance_loss + block_balance
            observations.append(observation)
        logits = self.head(self.final_norm(x))
        return logits, balance_loss / len(self.blocks), observations

    @torch.no_grad()
    def routes(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        _, _, observations = self(tokens)
        return [observation.executed_routes.clone() for observation in observations]

    @torch.no_grad()
    def busiest_first_layer_expert(self, tokens: torch.Tensor) -> int:
        _, _, observations = self(tokens)
        return int(observations[0].loads.argmax().item())

    @torch.no_grad()
    def shock_first_layer_expert(self, expert_index: int, scale: float) -> None:
        self.blocks[0].moe.experts[expert_index].down.weight.mul_(scale)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(config: Config, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic multi-domain Markov sequences for next-token prediction."""
    generator = torch.Generator().manual_seed(seed + 10_000)
    increments = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(config.steps + config.delay):
        domains = torch.randint(0, config.num_domains, (config.batch_size,), generator=generator)
        values = torch.randint(0, 32, (config.batch_size,), generator=generator)
        sequence = torch.empty(config.batch_size, config.sequence_length + 1, dtype=torch.long)
        sequence[:, 0] = 32 + domains
        for position in range(1, config.sequence_length + 1):
            noise_mask = torch.rand(config.batch_size, generator=generator) < 0.05
            noise = torch.randint(0, 32, (config.batch_size,), generator=generator)
            values = (values + increments[domains]) % 32
            values = torch.where(noise_mask, noise, values)
            sequence[:, position] = values
        batches.append((sequence[:, :-1], sequence[:, 1:]))
    return batches


def route_disagreement(observations: list[MoEObservation]) -> float:
    disagreements = []
    for observation in observations:
        executed = observation.executed_routes.sort(dim=-1).values
        synchronous = observation.synchronous_routes.sort(dim=-1).values
        disagreements.append((executed != synchronous).any(dim=-1).float().mean())
    return float(torch.stack(disagreements).mean().item())


def gradient_norm(model: nn.Module) -> float:
    squared = torch.zeros(())
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().pow(2).sum().cpu()
    return float(torch.sqrt(squared).item())


def make_optimizer(model: nn.Module, config: Config) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def train_step(
    model: TinyMoELanguageModel,
    optimizer: torch.optim.AdamW,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    forced_routes: list[torch.Tensor] | None,
    step: int,
    target_expert: int,
    config: Config,
) -> StepObservation:
    logits, balance_loss, layer_observations = model(tokens, forced_routes)
    task_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss = task_loss + config.balance_weight * balance_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = gradient_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    margin = torch.cat([item.margin.flatten() for item in layer_observations]).mean()
    maximum = torch.stack([item.max_expert_output for item in layer_observations]).max()
    first_target_load = (
        0.0 if target_expert < 0 else float(layer_observations[0].loads[target_expert].item())
    )
    return StepObservation(
        step=step,
        loss=float(loss.detach().item()),
        task_loss=float(task_loss.detach().item()),
        balance_loss=float(balance_loss.detach().item()),
        route_disagreement=route_disagreement(layer_observations),
        route_margin=float(margin.item()),
        max_expert_output=float(maximum.item()),
        first_layer_target_load=first_target_load,
        gradient_norm=grad_norm,
    )


def summarize_run(
    condition: Condition,
    seed: int,
    target_expert: int,
    observations: list[StepObservation],
    config: Config,
) -> dict[str, object]:
    losses = np.asarray([observation.task_loss for observation in observations])
    baseline_start = max(0, config.shock_step - 20)
    baseline = float(losses[baseline_start : config.shock_step].mean())
    post = losses[config.shock_step : config.shock_step + config.recovery_steps]
    excess = np.maximum(post - baseline, 0.0)
    smoothed = np.convolve(post, np.ones(5) / 5, mode="valid")
    threshold = baseline + max(0.02, 0.05 * baseline)
    recovered = np.flatnonzero(smoothed <= threshold)
    recovery_time = None if recovered.size == 0 else int(recovered[0] + 4)
    return {
        "condition": condition,
        "seed": seed,
        "target_expert": target_expert,
        "baseline_loss": baseline,
        "max_post_shock_loss": float(post.max()),
        "positive_excess_loss_auc": float(excess.sum()),
        "recovery_time_steps": recovery_time,
        "mean_route_disagreement": float(
            np.mean(
                [
                    observation.route_disagreement
                    for observation in observations[
                        config.shock_step : config.shock_step + config.recovery_steps
                    ]
                ]
            )
        ),
        "peak_expert_output": float(
            max(
                observation.max_expert_output
                for observation in observations[
                    config.shock_step : config.shock_step + config.recovery_steps
                ]
            )
        ),
        "observations": [asdict(observation) for observation in observations],
    }


def pretrain_to_shock(
    config: Config,
    seed: int,
    device: torch.device,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, object],
    dict[int, list[torch.Tensor]],
    list[StepObservation],
    int,
]:
    seed_everything(seed)
    model = TinyMoELanguageModel(config).to(device)
    optimizer = make_optimizer(model, config)

    historical_routes: dict[int, list[torch.Tensor]] = {}
    for step in range(config.delay):
        future_tokens = batches[step][0].to(device)
        historical_routes[step] = model.routes(future_tokens)

    observations: list[StepObservation] = []
    model.train()

    for step in range(config.shock_step):
        tokens, targets = batches[step]
        tokens, targets = tokens.to(device), targets.to(device)

        future_step = step + config.delay
        if future_step < len(batches):
            future_tokens = batches[future_step][0].to(device)
            historical_routes[future_step] = model.routes(future_tokens)

        observations.append(
            train_step(
                model,
                optimizer,
                tokens,
                targets,
                forced_routes=None,
                step=step,
                target_expert=-1,
                config=config,
            )
        )

    shock_tokens = batches[config.shock_step][0].to(device)
    target_expert = model.busiest_first_layer_expert(shock_tokens)
    return (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        historical_routes,
        observations,
        target_expert,
    )


def run_condition(
    config: Config,
    condition: Condition,
    seed: int,
    device: torch.device,
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, object],
    initial_historical_routes: dict[int, list[torch.Tensor]],
    pre_shock_observations: list[StepObservation],
    target_expert: int,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    seed_everything(seed)
    model = TinyMoELanguageModel(config).to(device)
    model.load_state_dict(model_state)
    optimizer = make_optimizer(model, config)
    # ``load_state_dict`` may retain references to optimizer-state tensors.  Each branch must get
    # an immutable copy or the first recovery condition will mutate the checkpoint used by the next.
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    historical_routes = copy.deepcopy(initial_historical_routes)
    observations = copy.deepcopy(pre_shock_observations)
    model.shock_first_layer_expert(target_expert, config.shock_scale)
    model.train()

    for step in range(config.shock_step, config.steps):
        tokens, targets = batches[step]
        tokens, targets = tokens.to(device), targets.to(device)

        future_step = step + config.delay
        if future_step < len(batches):
            future_tokens = batches[future_step][0].to(device)
            historical_routes[future_step] = model.routes(future_tokens)

        historical_active = (
            condition == "historical"
            and config.shock_step <= step < config.shock_step + config.recovery_steps
        )
        forced_routes = historical_routes[step] if historical_active else None
        observations.append(
            train_step(
                model,
                optimizer,
                tokens,
                targets,
                forced_routes,
                step=step,
                target_expert=target_expert,
                config=config,
            )
        )

    return summarize_run(condition, seed, target_expert, observations, config)


def paired_effects(results: list[dict[str, object]]) -> dict[str, object]:
    """Summarize paired historical-minus-synchronous effects."""
    by_seed: dict[int, dict[str, dict[str, object]]] = {}
    for result in results:
        seed = int(result["seed"])
        condition = str(result["condition"])
        by_seed.setdefault(seed, {})[condition] = result

    effects: list[dict[str, float | int]] = []
    for seed, conditions in sorted(by_seed.items()):
        synchronous = conditions["synchronous"]
        historical = conditions["historical"]
        synchronous_recovery = synchronous["recovery_time_steps"]
        historical_recovery = historical["recovery_time_steps"]
        effect: dict[str, float | int] = {
            "seed": seed,
            "max_post_shock_loss_delta": float(historical["max_post_shock_loss"])
            - float(synchronous["max_post_shock_loss"]),
            "positive_excess_loss_auc_delta": float(historical["positive_excess_loss_auc"])
            - float(synchronous["positive_excess_loss_auc"]),
        }
        if synchronous_recovery is not None and historical_recovery is not None:
            effect["recovery_time_steps_delta"] = int(historical_recovery) - int(
                synchronous_recovery
            )
        effects.append(effect)

    metric_names = [
        "max_post_shock_loss_delta",
        "positive_excess_loss_auc_delta",
        "recovery_time_steps_delta",
    ]
    means = {
        metric: float(np.mean([effect[metric] for effect in effects if metric in effect]))
        for metric in metric_names
        if any(metric in effect for effect in effects)
    }
    seed_count = len(effects)
    return {
        "definition": "historical minus synchronous; negative favors historical",
        "per_seed": effects,
        "means": means,
        "warning": (
            f"{seed_count} synthetic seed{'s' if seed_count != 1 else ''} "
            f"{'are' if seed_count != 1 else 'is'} a mechanism probe, not "
            "confirmatory evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated integer seeds")
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--shock-step", type=int, default=Config.shock_step)
    parser.add_argument("--recovery-steps", type=int, default=Config.recovery_steps)
    parser.add_argument("--shock-scale", type=float, default=Config.shock_scale)
    parser.add_argument("--delay", type=int, default=Config.delay)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = Config(
        steps=args.steps,
        shock_step=args.shock_step,
        recovery_steps=args.recovery_steps,
        shock_scale=args.shock_scale,
        delay=args.delay,
    )
    config.validate()
    device = choose_device(args.device)
    seeds = [int(item) for item in args.seeds.split(",")]
    results: list[dict[str, object]] = []

    for seed in seeds:
        batches = make_batches(config, seed)
        (
            model_state,
            optimizer_state,
            historical_routes,
            pre_shock_observations,
            target_expert,
        ) = pretrain_to_shock(config, seed, device, batches)
        for condition in ("synchronous", "historical"):
            result = run_condition(
                config,
                condition,
                seed,
                device,
                model_state,
                optimizer_state,
                historical_routes,
                pre_shock_observations,
                target_expert,
                batches,
            )
            results.append(result)
            print(
                condition,
                "seed=",
                seed,
                "auc=",
                f"{result['positive_excess_loss_auc']:.4f}",
                "recovery=",
                result["recovery_time_steps"],
                "route_disagreement=",
                f"{result['mean_route_disagreement']:.4f}",
            )

    payload = {
        "warning": "Synthetic mechanism probe; not evidence of language-model-scale benefit.",
        "device": str(device),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "results": results,
        "paired_effects": paired_effects(results),
    }
    serialized = json.dumps(payload, indent=2)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
