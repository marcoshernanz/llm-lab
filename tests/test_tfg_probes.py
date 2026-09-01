from __future__ import annotations

import pytest
import torch

from experiments.tfg import anticipatory_routing_probe as routing_probe
from experiments.tfg import recurrent_state_quantization_probe as state_probe


def _task_losses(result: dict[str, object]) -> list[float]:
    observations = result["observations"]
    assert isinstance(observations, list)
    return [float(row["task_loss"]) for row in observations]


def test_routing_recovery_branches_do_not_share_optimizer_mutations() -> None:
    config = routing_probe.Config(
        steps=14,
        shock_step=7,
        delay=2,
        recovery_steps=5,
        batch_size=4,
        sequence_length=8,
        model_dim=16,
        num_heads=2,
        num_blocks=2,
        expert_hidden_dim=24,
    )
    batches = routing_probe.make_batches(config, seed=0)
    checkpoint = routing_probe.pretrain_to_shock(
        config, seed=0, device=torch.device("cpu"), batches=batches
    )
    model_state, optimizer_state, routes, observations, target_expert = checkpoint

    def run(condition: routing_probe.Condition) -> dict[str, object]:
        return routing_probe.run_condition(
            config,
            condition,
            seed=0,
            device=torch.device("cpu"),
            model_state=model_state,
            optimizer_state=optimizer_state,
            initial_historical_routes=routes,
            pre_shock_observations=observations,
            target_expert=target_expert,
            batches=batches,
        )

    synchronous_first = run("synchronous")
    historical_second = run("historical")
    historical_first = run("historical")
    synchronous_second = run("synchronous")

    assert _task_losses(synchronous_first) == pytest.approx(
        _task_losses(synchronous_second), abs=1e-10
    )
    assert _task_losses(historical_second) == pytest.approx(
        _task_losses(historical_first), abs=1e-10
    )


def test_uniform_int8_recurrent_state_is_less_destructive_than_int4() -> None:
    config = state_probe.Config(sequence_lengths=(128,), decays=(0.999,), seeds=(0,))
    int4 = state_probe.recurrent_trial(seed=0, steps=128, decay=0.999, bits=4, config=config)
    int8 = state_probe.recurrent_trial(seed=0, steps=128, decay=0.999, bits=8, config=config)

    assert float(int8["state_relative_l2"]) < float(int4["state_relative_l2"])
    assert float(int8["trajectory_relative_l2"]) < float(int4["trajectory_relative_l2"])
