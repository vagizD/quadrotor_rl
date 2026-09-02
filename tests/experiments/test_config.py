from dataclasses import replace

import numpy as np
import pytest

from common.dimensions import QDIMS
from experiments import (
    ExperimentConfig,
    EvaluationConfig,
    HoverEnvironmentConfig,
    DimensionsConfig,
    PPOConfig,
    RewardConfig,
    SimulationConfig,
    TrackingConfig,
    TrainingConfig,
    load_experiment_config,
    save_resolved_config,
)


CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def test_baseline_config_loads_into_validated_objects() -> None:
    config = load_experiment_config(CONFIG_PATH)

    assert isinstance(config, ExperimentConfig)
    assert config.name == "ppo_hover_baseline"
    assert config.seed == 42
    assert config.algorithm == "ppo"
    assert config.simulation == SimulationConfig(dt=0.01, integrator="rk4")
    assert isinstance(config.environment, HoverEnvironmentConfig)
    assert isinstance(config.dimensions, DimensionsConfig)
    assert config.dimensions.observation_dim == 14
    assert config.dimensions.action_dim == QDIMS.action_dim
    assert isinstance(config.reward, RewardConfig)
    assert isinstance(config.ppo, PPOConfig)
    assert isinstance(config.tracking, TrackingConfig)
    assert config.tracking.console_log_interval == 1
    assert config.tracking.rerun_step_stride == 10
    assert not config.tracking.record_evaluation_rerun
    assert isinstance(config.evaluation, EvaluationConfig)
    assert config.evaluation.interval_updates == 10
    assert config.evaluation.episodes == 2
    assert config.evaluation.seed == 1042
    assert config.environment.success.position_error == 0.10
    assert config.environment.success.speed == 0.25
    assert config.environment.success.angular_rate == 0.25
    assert config.environment.success.tilt_angle == np.deg2rad(15.0)
    assert config.environment.success.heading_error == np.deg2rad(15.0)
    assert isinstance(config.training, TrainingConfig)
    assert config.training.updates == 1000
    assert config.training.rolling_episode_window == 100
    assert config.ppo.hidden_sizes == (64, 64)
    assert config.ppo.initial_log_std == -1.5
    assert config.ppo.min_log_std == -2.0
    assert config.ppo.max_log_std == -1.0
    assert config.ppo.gamma == 0.999
    assert config.ppo.gae_lambda == 0.95
    assert config.ppo.clip_epsilon == 0.2
    assert config.ppo.learning_rate == 0.0003
    assert config.ppo.value_loss_coef == 0.5
    assert config.ppo.entropy_coef == 0.0
    assert config.ppo.rollout_steps == 2048
    assert config.ppo.steps_per_rollout == 4
    assert config.ppo.normalize_observations
    assert config.ppo.observation_clip == 10.0
    assert config.ppo.normalize_advantages
    assert config.ppo.normalization_epsilon == 1e-8
    assert config.ppo.reward_scale == 0.1
    assert config.ppo.max_grad_norm == 0.5
    np.testing.assert_allclose(config.environment.target_position, [0.0, 0.0, 0.0])
    assert config.environment.target_heading == 0.0
    assert config.environment.episode_horizon == 10.0
    assert config.environment.initializer.name == "random"
    assert config.environment.initializer.max_position_distance == 0.25
    assert config.environment.initializer.max_tilt_angle == np.deg2rad(15.0)
    assert config.reward.alive_reward == 1.0
    assert config.reward.position_error_weight == 4.0
    assert config.reward.heading_weight == 0.1
    assert config.reward.thrust_deviation_weight == 0.1
    assert config.reward.terminal_penalty == 10.0
    assert config.reward.truncation_distance_scale == 0.25
    assert config.reward.cost_normalization_scale == 5.0
    assert config.reward.normalized_cost_limit == 0.9
    assert config.quadrotor.mass == 1.0
    np.testing.assert_allclose(config.quadrotor.inertia, np.diag([0.01, 0.01, 0.02]))
    np.testing.assert_allclose(
        config.quadrotor.motor_spin_directions,
        [1.0, -1.0, 1.0, -1.0],
    )


def test_config_rejects_unknown_sections(tmp_path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text("[environment]\nname = 'hover'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        load_experiment_config(path)


def test_resolved_config_snapshot_round_trips(tmp_path) -> None:
    config = load_experiment_config(CONFIG_PATH)
    snapshot_path = tmp_path / "run" / "config.toml"

    assert save_resolved_config(config, snapshot_path) == snapshot_path
    restored = load_experiment_config(snapshot_path)

    assert restored.as_dict() == config.as_dict()


def test_config_requires_all_current_sections(tmp_path) -> None:
    path = tmp_path / "minimal.toml"
    path.write_text(
        """[quadrotor]
mass = 1.0
arm_length = 0.2
inertia = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.02]]
yaw_torque_coefficient = 0.1
max_thrust = 5.0

[simulation]
dt = 0.01
integrator = "euler"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly"):
        load_experiment_config(path)


def test_config_rejects_dimensions_incompatible_with_current_model() -> None:
    config = load_experiment_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="observation_dim"):
        replace(
            config,
            dimensions=DimensionsConfig(
                observation_dim=config.dimensions.observation_dim + 1,
                action_dim=config.dimensions.action_dim,
            ),
        )


def test_reward_config_requires_positive_normalization_scales() -> None:
    config = load_experiment_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="cost_normalization_scale"):
        replace(
            config,
            reward=replace(config.reward, cost_normalization_scale=0.0),
        )

    with pytest.raises(ValueError, match="truncation_distance_scale"):
        replace(
            config,
            reward=replace(config.reward, truncation_distance_scale=0.0),
        )

    with pytest.raises(ValueError, match="normalized_cost_limit"):
        replace(
            config,
            reward=replace(
                config.reward,
                normalized_cost_limit=config.reward.alive_reward,
            ),
        )
