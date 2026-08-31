from dataclasses import replace

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.dimensions import QDIMS
from environments import HoverEnvironment
from experiments import load_experiment_config
from rl.ppo import Actor, Critic, PPOTrajectory, collect_ppo_trajectory
from simulation import Trajectory
from visualization.plotting import plot_trajectory, save_trajectory_plot


CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def make_components(
    episode_horizon: float | None = None,
) -> tuple[HoverEnvironment, Actor, Critic]:
    config = load_experiment_config(CONFIG_PATH)
    initializer = replace(
        config.environment.initializer,
        max_position_distance=0.0,
        max_velocity=0.0,
        max_tilt_angle=0.0,
        max_angular_velocity=0.0,
    )
    environment_config = replace(
        config.environment,
        initializer=initializer,
        **(
            {} if episode_horizon is None else {"episode_horizon": episode_horizon}
        ),
    )
    config = replace(config, environment=environment_config)
    environment = HoverEnvironment.from_config(config)
    actor = Actor(
        config.dimensions.observation_dim,
        config.dimensions.action_dim,
        config.ppo.hidden_sizes,
        config.ppo.initial_log_std,
        config.ppo.min_log_std,
        config.ppo.max_log_std,
    )
    critic = Critic(
        config.dimensions.observation_dim,
        config.ppo.hidden_sizes,
    )
    return environment, actor, critic


def test_ppo_collection_aligns_policy_and_physics_samples() -> None:
    environment, actor, critic = make_components()
    steps = 3
    torch.manual_seed(7)

    trajectory = collect_ppo_trajectory(environment, actor, critic, steps)

    assert isinstance(trajectory, PPOTrajectory)
    assert isinstance(trajectory, Trajectory)
    assert trajectory.step_count == steps
    assert trajectory.times.shape == (steps + 1,)
    assert trajectory.states.shape == (steps + 1, QDIMS.state_dim)
    assert trajectory.actions.shape == (steps, QDIMS.action_dim)
    assert trajectory.observations.shape == (
        steps,
        environment.observation_dim,
    )
    for array in (
        trajectory.rewards,
        trajectory.values,
        trajectory.log_probabilities,
        trajectory.raw_costs,
        trajectory.normalized_costs,
        trajectory.terminated,
        trajectory.truncated,
    ):
        assert array.shape == (steps,)
    assert trajectory.terminated.dtype == bool
    assert trajectory.truncated.dtype == bool
    assert not trajectory.terminated.any()
    assert not trajectory.truncated.any()
    np.testing.assert_allclose(
        trajectory.times,
        np.arange(steps + 1, dtype=np.float64) * environment.integrator.dt,
    )
    np.testing.assert_allclose(trajectory.actions, trajectory.thrusts)
    np.testing.assert_allclose(
        trajectory.target_positions,
        environment.target_position,
    )

    figure = plot_trajectory(trajectory)
    assert len(figure.axes) == QDIMS.state_dim + QDIMS.action_dim
    plt.close(figure)


def test_ppo_trajectory_inherits_serialization(tmp_path) -> None:
    environment, actor, critic = make_components()
    torch.manual_seed(11)
    trajectory = collect_ppo_trajectory(environment, actor, critic, steps=2)
    trajectory.compute_returns(gamma=0.99)
    trajectory.compute_gae(gamma=0.99, gae_lambda=0.95)
    path = tmp_path / "ppo_trajectory.npz"

    trajectory.save(path)
    restored = PPOTrajectory.load(path)

    assert isinstance(restored, PPOTrajectory)
    np.testing.assert_allclose(restored.times, trajectory.times)
    np.testing.assert_allclose(restored.states, trajectory.states)
    np.testing.assert_allclose(restored.thrusts, trajectory.thrusts)
    np.testing.assert_allclose(restored.observations, trajectory.observations)
    np.testing.assert_allclose(restored.rewards, trajectory.rewards)
    np.testing.assert_allclose(restored.raw_costs, trajectory.raw_costs)
    np.testing.assert_allclose(
        restored.normalized_costs,
        trajectory.normalized_costs,
    )
    np.testing.assert_allclose(restored.values, trajectory.values)
    np.testing.assert_allclose(
        restored.log_probabilities,
        trajectory.log_probabilities,
    )
    np.testing.assert_array_equal(restored.terminated, trajectory.terminated)
    np.testing.assert_array_equal(restored.truncated, trajectory.truncated)
    assert restored.boundary_state_value == trajectory.boundary_state_value
    np.testing.assert_allclose(restored.returns, trajectory.returns)
    np.testing.assert_allclose(restored.advantages, trajectory.advantages)

    plot_path = tmp_path / "ppo_trajectory.png"
    assert save_trajectory_plot(restored, plot_path) == plot_path
    assert plot_path.exists() and plot_path.stat().st_size > 0


def test_ppo_collection_stops_at_first_episode_boundary() -> None:
    environment, actor, critic = make_components(episode_horizon=0.015)
    torch.manual_seed(13)

    trajectory = collect_ppo_trajectory(environment, actor, critic, steps=5)

    assert trajectory.step_count == 2
    assert not trajectory.terminated.any()
    assert trajectory.truncated[-1]
    assert not trajectory.truncated[:-1].any()
