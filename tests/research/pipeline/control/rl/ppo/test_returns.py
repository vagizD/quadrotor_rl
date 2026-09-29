import numpy as np

from geometry.common.dimensions import QDIMS
from research.pipeline.control.rl.ppo import PPOTrajectory


def make_trajectory(
    rewards: list[float],
    values: list[float],
    terminated: list[bool] | None = None,
    truncated: list[bool] | None = None,
    boundary_state_value: float = 0.0,
) -> PPOTrajectory:
    step_count = len(rewards)
    return PPOTrajectory(
        times=np.arange(step_count + 1, dtype=np.float64),
        states=np.zeros((step_count + 1, QDIMS.state_dim)),
        thrusts=np.zeros((step_count, QDIMS.action_dim)),
        observations=np.zeros((step_count, QDIMS.observation_dim)),
        rewards=np.asarray(rewards, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        log_probabilities=np.zeros(step_count),
        terminated=np.zeros(step_count, dtype=bool)
        if terminated is None
        else np.asarray(terminated, dtype=bool),
        truncated=np.zeros(step_count, dtype=bool)
        if truncated is None
        else np.asarray(truncated, dtype=bool),
        boundary_state_value=boundary_state_value,
    )


def test_returns_discount_rewards_and_bootstrap_the_final_state() -> None:
    trajectory = make_trajectory(
        rewards=[1.0, 2.0, 3.0],
        values=[0.5, 1.0, 1.5],
        truncated=[False, False, True],
        boundary_state_value=4.0,
    )

    returns = trajectory.compute_returns(gamma=0.5)

    # G_2 = 3 + 0.5 * 4 = 5; G_1 = 2 + 0.5 * 5 = 4.5;
    # G_0 = 1 + 0.5 * 4.5 = 3.25.
    np.testing.assert_allclose(returns, [3.25, 4.5, 5.0])
    np.testing.assert_allclose(trajectory.returns, returns)


def test_returns_stop_bootstrapping_after_physical_termination() -> None:
    trajectory = make_trajectory(
        rewards=[1.0, 2.0],
        values=[0.0, 0.0],
        terminated=[False, True],
        boundary_state_value=100.0,
    )

    returns = trajectory.compute_returns(gamma=0.9)

    np.testing.assert_allclose(returns, [2.8, 2.0])


def test_reward_scale_changes_return_targets_only() -> None:
    trajectory = make_trajectory(
        rewards=[1.0, 2.0],
        values=[0.0, 0.0],
    )

    returns = trajectory.compute_returns(gamma=0.5, reward_scale=0.1)

    np.testing.assert_allclose(returns, [0.2, 0.2])


def test_gae_with_zero_lambda_is_one_step_td_error() -> None:
    trajectory = make_trajectory(
        rewards=[1.0, 2.0, 3.0],
        values=[0.5, 1.0, 1.5],
        boundary_state_value=2.0,
    )

    advantages = trajectory.compute_gae(gamma=0.5, gae_lambda=0.0)

    # delta_t = r_t + gamma * V_{t+1} - V_t.
    np.testing.assert_allclose(advantages, [1.0, 1.75, 2.5])


def test_gae_with_one_lambda_is_return_minus_value() -> None:
    trajectory = make_trajectory(
        rewards=[1.0, 2.0, 3.0],
        values=[0.5, 1.0, 1.5],
        boundary_state_value=2.0,
    )

    advantages = trajectory.compute_gae(gamma=0.5, gae_lambda=1.0)

    # With lambda=1, the TD residuals telescope to G_t - V_t.
    np.testing.assert_allclose(advantages, [2.5, 3.0, 2.5])


def test_gae_masks_the_value_after_physical_termination() -> None:
    trajectory = make_trajectory(
        rewards=[1.0, 2.0],
        values=[0.0, 0.0],
        terminated=[False, True],
        boundary_state_value=100.0,
    )

    advantages = trajectory.compute_gae(gamma=0.9, gae_lambda=0.95)

    np.testing.assert_allclose(advantages, [1.0 + 0.9 * 0.95 * 2.0, 2.0])
