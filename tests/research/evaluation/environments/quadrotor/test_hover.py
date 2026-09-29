from dataclasses import replace

import numpy as np
import pytest

from geometry.common.dimensions import QDIMS
from environments import HoverEnvironment
from research.pipeline.control.residual.models.experiments import load_experiment_config
from geometry.math.rotation import Quaternion
from robots.quadrotor import RigidBodyState
from research.simulator.integrators import EulerIntegrator, RK4Integrator

CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def make_environment(
    target_position: np.ndarray | None = None,
    integrator_type: type[EulerIntegrator] | type[RK4Integrator] = EulerIntegrator,
    episode_horizon: float | None = None,
    dt: float = 0.01,
    randomize: bool = False,
) -> HoverEnvironment:
    config = load_experiment_config(CONFIG_PATH)
    environment_config = config.environment
    if not randomize:
        initializer = replace(
            environment_config.initializer,
            max_position_distance=0.0,
            max_velocity=0.0,
            max_tilt_angle=0.0,
            max_angular_velocity=0.0,
        )
        environment_config = replace(
            environment_config,
            initializer=initializer,
        )
    if target_position is not None:
        environment_config = replace(
            environment_config,
            target_position=target_position,
        )
    if episode_horizon is not None:
        environment_config = replace(
            environment_config,
            episode_horizon=episode_horizon,
        )
    simulation_config = replace(
        config.simulation,
        dt=dt,
        integrator="euler" if integrator_type is EulerIntegrator else "rk4",
    )
    return HoverEnvironment.from_config(
        replace(
            config,
            environment=environment_config,
            simulation=simulation_config,
        ),
    )


def test_reset_restores_nominal_hover_state_and_time() -> None:
    environment = make_environment(np.array([1.0, 2.0, 3.0]))
    environment.episode_time = 4.0
    environment.quadrotor.state.position[:] = [9.0, 9.0, 9.0]
    environment.quadrotor.set_thrusts(np.zeros(environment.action_dim))

    state = environment.reset()

    np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.velocity, np.zeros(3))
    np.testing.assert_allclose(state.quaternion.as_array(), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.angular_velocity, np.zeros(3))
    np.testing.assert_allclose(
        environment.quadrotor.thrusts,
        np.full(
            environment.action_dim,
            environment.params.mass * environment.params.gravity / environment.action_dim,
        ),
    )
    assert environment.episode_time == 0.0
    np.testing.assert_allclose(
        environment.quadrotor.compute_linear_acceleration(),
        np.zeros(3),
    )


def test_reset_samples_configured_initial_state_reproducibly() -> None:
    first_environment = make_environment(randomize=True)
    second_environment = make_environment(randomize=True)

    first_state = first_environment.reset()
    second_state = second_environment.reset()
    initializer = first_environment.environment_config.initializer

    np.testing.assert_allclose(first_state.to_vector(), second_state.to_vector())
    assert initializer.min_position_distance <= np.linalg.norm(
        first_state.position - first_environment.target_position
    ) <= initializer.max_position_distance
    assert initializer.min_velocity <= np.linalg.norm(first_state.velocity)
    assert np.linalg.norm(first_state.velocity) <= initializer.max_velocity
    assert initializer.min_angular_velocity <= np.linalg.norm(
        first_state.angular_velocity
    )
    assert (
        np.linalg.norm(first_state.angular_velocity)
        <= initializer.max_angular_velocity
    )
    body_up_world = first_state.quaternion.to_rotation_matrix()[:, 2]
    tilt_angle = np.arccos(np.clip(body_up_world[2], -1.0, 1.0))
    assert initializer.min_tilt_angle <= tilt_angle <= initializer.max_tilt_angle
    assert first_environment.episode_time == 0.0


def test_target_is_copied_and_validated() -> None:
    target = np.array([1.0, 2.0, 3.0])
    environment = make_environment(target)
    target[0] = 99.0

    np.testing.assert_allclose(environment.target_position, [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="target_position"):
        make_environment(np.zeros(2))


def test_observation_has_frozen_order_and_target_to_state_error() -> None:
    environment = make_environment(np.array([10.0, 20.0, 30.0]))
    environment.quadrotor.state = RigidBodyState(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([4.0, 5.0, 6.0]),
        quaternion=Quaternion(0.5, 0.5, 0.5, 0.5),
        angular_velocity=np.array([7.0, 8.0, 9.0]),
    )

    observation = environment.get_observation()

    np.testing.assert_allclose(
        observation,
        [
            9.0,
            18.0,
            27.0,
            4.0,
            5.0,
            6.0,
            0.5,
            0.5,
            0.5,
            0.5,
            7.0,
            8.0,
            9.0,
            0.0,
        ],
        atol=1e-6,
    )
    assert observation.shape == (environment.observation_dim,)


@pytest.mark.parametrize(
    ("integrator_type", "expected_position_z"),
    [
        (EulerIntegrator, 0.0),
        (RK4Integrator, -0.5 * 9.81 * 0.1**2),
    ],
)
def test_step_advances_physics_and_time(
    integrator_type: type[EulerIntegrator] | type[RK4Integrator],
    expected_position_z: float,
) -> None:
    dt = 0.1
    environment = make_environment(
        np.array([0.0, 0.0, 1.0]),
        integrator_type,
        dt=dt,
    )

    observation, reward, terminated, truncated = environment.step(
        np.zeros(environment.action_dim)
    )

    assert environment.episode_time == dt
    assert reward < environment.reward_config.alive_reward
    assert not terminated
    assert not truncated
    np.testing.assert_allclose(environment.state.position[2], 1.0 + expected_position_z)
    np.testing.assert_allclose(environment.state.velocity, [0.0, 0.0, -9.81 * dt])
    np.testing.assert_allclose(observation[:3], [0.0, 0.0, -expected_position_z])


@pytest.mark.parametrize(
    "action",
    [
        np.full(QDIMS.motor_count, -0.1),
        np.full(QDIMS.motor_count, 5.1),
        np.ones(3),
        np.full(QDIMS.motor_count, np.nan),
    ],
)
def test_step_rejects_actions_outside_thrust_bounds(action: np.ndarray) -> None:
    environment = make_environment()

    with pytest.raises((ValueError, TypeError)):
        environment.step(action)

    assert environment.episode_time == 0.0


def test_nominal_hover_has_alive_reward_and_is_not_done() -> None:
    environment = make_environment()

    assert environment.compute_reward() == environment.reward_config.alive_reward
    assert not environment.is_terminated()


def test_reward_weight_comes_from_experiment_config() -> None:
    config = load_experiment_config(CONFIG_PATH)
    config = replace(
        config,
        reward=replace(config.reward, position_error_weight=2.0),
    )
    environment = HoverEnvironment.from_config(config)
    environment.state.position[0] = 1.0

    raw_cost = 2.0
    normalized_cost = config.reward.normalized_cost_limit * (-np.expm1(
        -raw_cost / config.reward.cost_normalization_scale
    ))
    np.testing.assert_allclose(
        environment.compute_reward(),
        config.reward.alive_reward
        - normalized_cost,
    )
    assert environment.last_raw_cost == raw_cost
    np.testing.assert_allclose(environment.last_normalized_cost, normalized_cost)


def test_target_heading_is_rewarded() -> None:
    config = load_experiment_config(CONFIG_PATH)
    environment = HoverEnvironment.from_config(
        replace(
            config,
            environment=replace(config.environment, target_heading=np.pi / 2.0),
        ),
    )

    assert environment.compute_reward() == environment.reward_config.alive_reward

    environment.quadrotor.state.quaternion = Quaternion.identity()

    raw_cost = environment.reward_config.heading_weight * (np.pi / 2.0) ** 2
    normalized_cost = environment.reward_config.normalized_cost_limit * (-np.expm1(
        -raw_cost / environment.reward_config.cost_normalization_scale
    ))
    expected_reward = (
        environment.reward_config.alive_reward
        - normalized_cost
    )
    np.testing.assert_allclose(environment.compute_reward(), expected_reward)


def test_terminal_penalty_applies_to_physical_failure_not_truncation() -> None:
    environment = make_environment(episode_horizon=0.1, dt=0.1)
    hover_thrust = (
        environment.params.mass
        * environment.params.gravity
        / environment.action_dim
    )

    _, truncated_reward, terminated, truncated = environment.step(
        np.full(environment.action_dim, hover_thrust)
    )

    assert not terminated
    assert truncated
    assert truncated_reward == environment.reward_config.alive_reward

    environment = make_environment()
    environment.state.position[0] = environment.max_position_error

    failure_reward = environment.compute_reward()

    assert environment.is_terminated()
    raw_cost = environment.reward_config.position_error_weight * environment.max_position_error
    normalized_cost = environment.reward_config.normalized_cost_limit * (-np.expm1(
        -raw_cost / environment.reward_config.cost_normalization_scale
    ))
    expected_reward = (
        environment.reward_config.alive_reward
        - normalized_cost
        - environment.reward_config.terminal_penalty
    )
    np.testing.assert_allclose(failure_reward, expected_reward)


def test_truncation_penalty_ranks_final_distance_without_failure() -> None:
    environment = make_environment(episode_horizon=0.1, dt=0.1)
    environment.state.position[0] = 1.0
    environment.episode_time = 0.1

    reward = environment.compute_reward()

    assert not environment.is_terminated()
    assert environment.is_truncated()
    dist = 1.0
    hover_gate = np.exp(-(dist**2) / (2.0 * (0.30**2)))
    transit_cost = environment.reward_config.position_error_weight * dist
    hover_cost = environment.reward_config.position_error_weight * (dist**2)
    raw_cost = (1.0 - hover_gate) * transit_cost + hover_gate * hover_cost
    normalized_cost = environment.reward_config.normalized_cost_limit * (-np.expm1(
        -raw_cost / environment.reward_config.cost_normalization_scale
    ))
    distance_score = 1.0 / (
        1.0 + 1.0 / environment.reward_config.truncation_distance_scale
    )
    expected_reward = (
        environment.reward_config.alive_reward
        - normalized_cost
        - environment.reward_config.terminal_penalty
        * (1.0 - distance_score)
    )
    np.testing.assert_allclose(reward, expected_reward)


def test_cost_normalization_is_bounded_and_exposes_raw_cost() -> None:
    environment = make_environment()
    environment.state.position[0] = 2.0

    reward = environment.compute_reward()

    assert not environment.is_terminated()
    dist = 2.0
    hover_gate = np.exp(-(dist**2) / (2.0 * (0.30**2)))
    transit_cost = environment.reward_config.position_error_weight * dist
    hover_cost = environment.reward_config.position_error_weight * (dist**2)
    raw_cost = (1.0 - hover_gate) * transit_cost + hover_gate * hover_cost
    expected_normalized_cost = environment.reward_config.normalized_cost_limit * (
        -np.expm1(
            -raw_cost / environment.reward_config.cost_normalization_scale
        )
    )
    expected_reward = (
        environment.reward_config.alive_reward
        - expected_normalized_cost
    )
    np.testing.assert_allclose(reward, expected_reward)
    assert environment.last_raw_cost == raw_cost
    np.testing.assert_allclose(
        environment.last_normalized_cost,
        expected_normalized_cost,
    )
    assert 0.0 <= environment.last_normalized_cost < (
        environment.reward_config.normalized_cost_limit
    )


def test_cost_normalization_preserves_order_without_a_hard_plateau() -> None:
    environment = make_environment()

    environment.state.position[0] = 2.0
    environment.compute_reward()
    normalized_cost_at_two = environment.last_normalized_cost

    environment.state.position[0] = 3.0
    environment.compute_reward()
    normalized_cost_at_three = environment.last_normalized_cost

    assert normalized_cost_at_three > normalized_cost_at_two


def test_step_returns_reward_and_separate_episode_flags() -> None:
    environment = make_environment()
    hover_thrust = (
        environment.params.mass
        * environment.params.gravity
        / environment.action_dim
    )

    observation, reward, terminated, truncated = environment.step(
        np.full(environment.action_dim, hover_thrust)
    )

    np.testing.assert_allclose(observation, environment.get_observation())
    assert reward == environment.reward_config.alive_reward
    assert not terminated
    assert not truncated


def test_horizon_terminates_after_fixed_simulation_time() -> None:
    environment = make_environment(episode_horizon=0.2, dt=0.1)
    hover_thrust = (
        environment.params.mass
        * environment.params.gravity
        / environment.action_dim
    )

    _, _, first_terminated, first_truncated = environment.step(
        np.full(environment.action_dim, hover_thrust)
    )
    _, _, second_terminated, second_truncated = environment.step(
        np.full(environment.action_dim, hover_thrust)
    )

    assert not first_terminated
    assert not first_truncated
    assert not second_terminated
    assert second_truncated


def test_failure_bounds_terminate_large_position_error_and_tilt() -> None:
    position_environment = make_environment()
    position_environment.quadrotor.state.position[0] = 6.0
    assert position_environment.is_terminated()

    tilt_environment = make_environment()
    tilt_environment.quadrotor.state = RigidBodyState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        quaternion=Quaternion(np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0),
        angular_velocity=np.zeros(3),
    )
    assert tilt_environment.is_terminated()


def test_velocity_and_angular_rate_are_penalized_but_recoverable() -> None:
    environment = make_environment()
    environment.quadrotor.state.velocity[:] = [20.0, 0.0, 0.0]
    environment.quadrotor.state.angular_velocity[:] = [20.0, 0.0, 0.0]

    assert not environment.is_terminated()
    assert environment.compute_reward() < environment.reward_config.alive_reward
