from dataclasses import replace

import numpy as np
import pytest
import torch

from environments import HoverEnvironment
from research.pipeline.control.residual.models.experiments import load_experiment_config
from research.pipeline.control.rl.ppo import Actor, DeterministicEvaluator


CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def make_evaluator() -> DeterministicEvaluator:
    config = load_experiment_config(CONFIG_PATH)
    initializer_config = replace(
        config.environment.initializer,
        min_position_distance=0.1,
        max_position_distance=0.1,
        min_velocity=0.1,
        max_velocity=0.1,
        min_tilt_angle=0.05,
        max_tilt_angle=0.05,
        min_angular_velocity=0.05,
        max_angular_velocity=0.05,
    )
    environment_config = replace(
        config.environment,
        initializer=initializer_config,
        episode_horizon=0.1,
        success=replace(
            config.environment.success,
            position_error=1.0,
            speed=1.0,
            angular_rate=1.0,
            tilt_angle=1.0,
            heading_error=1.0,
        ),
    )
    environment = HoverEnvironment.from_config(
        replace(config, environment=environment_config)
    )
    actor = Actor(
        config.dimensions.observation_dim,
        config.dimensions.action_dim,
        config.ppo.hidden_sizes,
        config.ppo.initial_log_std,
        config.ppo.min_log_std,
        config.ppo.max_log_std,
    )
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    evaluation_config = replace(config.evaluation, episodes=2)
    return DeterministicEvaluator(environment, actor, evaluation_config)


def test_deterministic_evaluation_reuses_fixed_initial_states() -> None:
    evaluator = make_evaluator()
    evaluator.actor.train()

    first = evaluator.evaluate()
    second = evaluator.evaluate()

    assert evaluator.actor.training
    assert len(first.episodes) == 2
    assert all(episode.truncated for episode in first.episodes)
    assert all(episode.success for episode in first.episodes)
    for first_episode, second_episode in zip(first.episodes, second.episodes):
        np.testing.assert_allclose(
            first_episode.trajectory.states,
            second_episode.trajectory.states,
        )
        np.testing.assert_allclose(
            first_episode.trajectory.thrusts,
            second_episode.trajectory.thrusts,
        )


def test_evaluation_summary_produces_training_metrics() -> None:
    summary = make_evaluator().evaluate()

    metrics = summary.as_metrics(update=10)

    assert metrics["update"] == 10
    assert metrics["evaluation_success_rate"] == 1.0
    assert metrics["evaluation_failure_rate"] == 0.0
    assert metrics["evaluation_truncation_rate"] == 1.0
    assert metrics["evaluation_rms_position_error"] >= 0.0


def test_evaluation_interval_is_configured() -> None:
    evaluator = make_evaluator()

    assert not evaluator.should_evaluate(0)
    assert not evaluator.should_evaluate(9)
    assert evaluator.should_evaluate(10)
    assert not evaluator.should_evaluate(11)


def test_evaluation_config_rejects_nonpositive_interval() -> None:
    config = load_experiment_config(CONFIG_PATH).evaluation

    with pytest.raises(ValueError, match="interval_updates"):
        replace(config, interval_updates=0)
