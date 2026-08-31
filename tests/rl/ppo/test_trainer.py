import math
from dataclasses import replace

import numpy as np
import torch

from experiments import load_experiment_config
from rl.ppo import (
    PPORollout,
    PPOTrainer,
    collect_ppo_rollout,
    collect_ppo_trajectory,
    compute_clipped_surrogate,
    compute_probability_ratio,
)
from tests.rl.ppo.test_rollout import make_components


CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def test_ppo_loop_computes_ratios_clips_and_updates_parameters() -> None:
    config = load_experiment_config(CONFIG_PATH)
    old_log_probabilities = torch.zeros(4)
    new_log_probabilities = torch.log(
        torch.tensor(
            [
                1.0 + 2.0 * config.ppo.clip_epsilon,
                1.0 - 2.0 * config.ppo.clip_epsilon,
                1.0,
                1.0 - 2.0 * config.ppo.clip_epsilon,
            ]
        )
    )
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])

    ratios = compute_probability_ratio(
        new_log_probabilities,
        old_log_probabilities,
    )
    surrogate = compute_clipped_surrogate(
        ratios,
        advantages,
        config.ppo.clip_epsilon,
    )

    expected_ratios = torch.tensor(
        [
            1.0 + 2.0 * config.ppo.clip_epsilon,
            1.0 - 2.0 * config.ppo.clip_epsilon,
            1.0,
            1.0 - 2.0 * config.ppo.clip_epsilon,
        ]
    )
    expected_surrogate = torch.tensor(
        [
            1.0 + config.ppo.clip_epsilon,
            1.0 - 2.0 * config.ppo.clip_epsilon,
            -1.0,
            -(1.0 - config.ppo.clip_epsilon),
        ]
    )
    assert torch.allclose(ratios, expected_ratios)
    assert torch.allclose(surrogate, expected_surrogate)

    environment, actor, critic = make_components()
    torch.manual_seed(23)
    trajectories = [
        collect_ppo_trajectory(environment, actor, critic, steps=4)
        for _ in range(2)
    ]
    rollout = PPORollout(trajectories)
    trainer = PPOTrainer(
        actor,
        critic,
        config.ppo,
        action_low=0.0,
        action_high=config.quadrotor.max_thrust,
    )
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]

    policy_terms = trainer.compute_policy_terms(rollout)
    metrics = trainer.update(rollout)
    batch = rollout.to_batch()

    assert rollout.trajectory_count == 2
    assert rollout.step_count == 8
    assert batch.step_count == rollout.step_count
    np.testing.assert_allclose(
        batch.returns,
        np.concatenate([trajectory.returns for trajectory in trajectories]),
    )
    assert torch.isfinite(policy_terms.ratios).all()
    assert torch.isfinite(policy_terms.clipped_ratios).all()
    assert 0.0 <= float(policy_terms.clip_fraction) <= 1.0
    assert all(math.isfinite(value) for value in metrics.__dict__.values())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, critic.parameters())
    )
    actor_optimizer_state = trainer.actor_optimizer.state[next(actor.parameters())]
    critic_optimizer_state = trainer.critic_optimizer.state[next(critic.parameters())]
    assert int(actor_optimizer_state["step"]) == config.ppo.steps_per_rollout
    assert int(critic_optimizer_state["step"]) == config.ppo.steps_per_rollout
    assert metrics.actor_grad_norm >= 0.0
    assert metrics.critic_grad_norm >= 0.0


def test_trainer_normalizes_advantages_across_the_combined_batch() -> None:
    config = load_experiment_config(CONFIG_PATH)
    environment, actor, critic = make_components()
    torch.manual_seed(31)
    rollout = PPORollout(
        [
            collect_ppo_trajectory(environment, actor, critic, steps=4),
            collect_ppo_trajectory(environment, actor, critic, steps=4),
        ]
    )
    trainer = PPOTrainer(
        actor,
        critic,
        config.ppo,
        action_low=0.0,
        action_high=config.quadrotor.max_thrust,
    )

    trainer._prepare_targets(rollout)
    advantages = trainer._batch_tensors(rollout.to_batch())[3]

    assert abs(float(advantages.mean())) < 1e-6
    np.testing.assert_allclose(
        float(advantages.std(unbiased=False)),
        1.0,
        atol=1e-5,
    )


def test_ppo_update_cycle_collects_configured_rollout() -> None:
    config = load_experiment_config(CONFIG_PATH)
    ppo_config = replace(config.ppo, rollout_steps=5)
    environment, actor, critic = make_components(episode_horizon=0.015)
    torch.manual_seed(29)

    rollout = collect_ppo_rollout(
        environment,
        actor,
        critic,
        steps=ppo_config.rollout_steps,
    )
    trainer = PPOTrainer(
        actor,
        critic,
        ppo_config,
        action_low=0.0,
        action_high=config.quadrotor.max_thrust,
    )
    metrics = trainer.update(rollout)

    assert rollout.step_count == ppo_config.rollout_steps
    assert rollout.trajectory_count == 3
    assert [trajectory.step_count for trajectory in rollout.trajectories] == [2, 2, 1]
    assert all(math.isfinite(value) for value in metrics.__dict__.values())
