"""Quadrotor hover task state and reset behavior."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from common.dimensions import QDIMS
from common.types import FloatVector
from experiments import (
    DimensionsConfig,
    ExperimentConfig,
    HoverEnvironmentConfig,
    RewardConfig,
)
from geometry.rotation import Quaternion
from .initialization import RandomInitializer
from robots.quadrotor import (
    Quadrotor,
    QuadrotorParams,
    QuadrotorState,
    compute_state_derivative_vector,
)
from simulation.integrators import EulerIntegrator, RK4Integrator

OBSERVATION_ORDER = (
    "position_error_x",
    "position_error_y",
    "position_error_z",
    "velocity_world_x",
    "velocity_world_y",
    "velocity_world_z",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "angular_velocity_body_x",
    "angular_velocity_body_y",
    "angular_velocity_body_z",
)


class HoverEnvironment:
    def __init__(
        self,
        params: QuadrotorParams,
        integrator: EulerIntegrator | RK4Integrator,
        dimensions: DimensionsConfig,
        environment_config: HoverEnvironmentConfig,
        reward_config: RewardConfig,
        initializer: RandomInitializer,
    ) -> None:
        if not isinstance(params, QuadrotorParams):
            raise TypeError("params must be QuadrotorParams")
        if not isinstance(integrator, (EulerIntegrator, RK4Integrator)):
            raise TypeError("integrator must be EulerIntegrator or RK4Integrator")
        if not isinstance(dimensions, DimensionsConfig):
            raise TypeError("dimensions must be DimensionsConfig")
        if dimensions.observation_dim != len(OBSERVATION_ORDER):
            raise ValueError(
                "dimensions observation_dim must match OBSERVATION_ORDER"
            )
        if dimensions.action_dim != params.motor_spin_directions.size:
            raise ValueError(
                "dimensions action_dim must match quadrotor motor count"
            )
        if not isinstance(environment_config, HoverEnvironmentConfig):
            raise TypeError("environment_config must be HoverEnvironmentConfig")
        if not isinstance(reward_config, RewardConfig):
            raise TypeError("reward_config must be RewardConfig")
        if not isinstance(initializer, RandomInitializer):
            raise TypeError("initializer must be RandomInitializer")
        self.params = params
        self.integrator = integrator
        self.dimensions = dimensions
        self.observation_dim = dimensions.observation_dim
        self.action_dim = dimensions.action_dim
        self.environment_config = environment_config
        self.reward_config = reward_config
        self.success_config = environment_config.success
        self.initializer = initializer
        self._target_position = environment_config.target_position.copy()
        self._target_heading = environment_config.target_heading
        self.episode_horizon = environment_config.episode_horizon
        self.max_position_error = environment_config.max_position_error
        self.max_tilt_angle = environment_config.max_tilt_angle
        self.episode_time = 0.0
        self._last_raw_cost = 0.0
        self._last_normalized_cost = 0.0
        self.quadrotor = Quadrotor(params, self._nominal_state())
        self._set_hover_thrust()

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "HoverEnvironment":
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be ExperimentConfig")
        integrator_type = (
            EulerIntegrator
            if config.simulation.integrator == "euler"
            else RK4Integrator
        )
        return cls(
            config.quadrotor,
            integrator_type(config.simulation.dt),
            config.dimensions,
            config.environment,
            config.reward,
            RandomInitializer(
                config.environment.initializer,
                config.environment.target_position,
                config.seed,
                target_heading=config.environment.target_heading,
            ),
        )

    @property
    def target_position(self) -> FloatVector:
        return self._target_position.copy()

    @property
    def target_heading(self) -> float:
        return self._target_heading

    @property
    def state(self) -> QuadrotorState:
        return self.quadrotor.state

    @property
    def last_raw_cost(self) -> float:
        return self._last_raw_cost

    @property
    def last_normalized_cost(self) -> float:
        return self._last_normalized_cost

    def get_observation(self) -> FloatVector:
        # This order is the fixed environment-to-policy interface.
        observation = np.concatenate(
            [
                self._target_position - self.state.position,
                self.state.velocity,
                self.state.quaternion.as_array(),
                self.state.angular_velocity,
            ]
        )
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                "hover observation shape does not match configured observation_dim"
            )
        return observation

    def step(self, action: ArrayLike) -> tuple[FloatVector, float, bool, bool]:
        """Apply one action and return observation, reward, and episode flags."""
        self.quadrotor.set_thrusts(action)
        thrusts = self.quadrotor.thrusts.copy()
        state_vector = self.state.to_vector()

        def derivative_function(current: FloatVector) -> FloatVector:
            return compute_state_derivative_vector(current, thrusts, self.params)

        next_state_vector = self.integrator.step(state_vector, derivative_function)
        self.quadrotor.state = QuadrotorState.from_vector(next_state_vector)
        self.episode_time += self.integrator.dt
        return (
            self.get_observation(),
            self.compute_reward(),
            self.is_terminated(),
            self.is_truncated(),
        )

    def compute_reward(self) -> float:
        """Return the current provisional hover reward."""
        position_error = self._target_position - self.state.position
        distance_to_target = np.linalg.norm(position_error)
        body_up_world = self.state.quaternion.to_rotation_matrix()[:, 2]
        heading_error = self._heading_error()
        hover_thrust = self.params.mass * self.params.gravity / QDIMS.motor_count
        thrust_deviation = self.quadrotor.thrusts - hover_thrust
        # Position and heading terms pursue the target; the remaining terms stabilize it.
        raw_cost = (
            self.reward_config.position_error_weight
            * np.dot(position_error, position_error)
            + self.reward_config.linear_velocity_weight
            * np.dot(self.state.velocity, self.state.velocity)
            + self.reward_config.tilt_weight * (1.0 - body_up_world[2])
            + self.reward_config.heading_weight * heading_error**2
            + self.reward_config.angular_velocity_weight * np.dot(
                self.state.angular_velocity,
                self.state.angular_velocity,
            )
            + self.reward_config.thrust_deviation_weight
            * np.dot(thrust_deviation, thrust_deviation)
        )
        # This monotone map uses the observed raw-cost scale without hard clipping.
        normalized_cost = self.reward_config.normalized_cost_limit * (-np.expm1(
            -raw_cost / self.reward_config.cost_normalization_scale
        ))
        self._last_raw_cost = float(raw_cost)
        self._last_normalized_cost = float(normalized_cost)
        reward = self.reward_config.alive_reward - normalized_cost
        if self.is_terminated():
            # Physical failure receives the full terminal penalty.
            reward -= self.reward_config.terminal_penalty
        elif self.is_truncated():
            distance_score = 1.0 / (
                1.0
                + distance_to_target
                / self.reward_config.truncation_distance_scale
            )
            # Rank horizon completions by final position without penalizing the target.
            reward -= self.reward_config.terminal_penalty * (
                1.0 - distance_score
            )
        return float(reward)

    def is_successful(self) -> bool:
        """Return whether the current state satisfies the hover tolerances."""
        if self.is_terminated():
            return False
        position_error = np.linalg.norm(
            self._target_position - self.state.position
        )
        speed = np.linalg.norm(self.state.velocity)
        angular_rate = np.linalg.norm(self.state.angular_velocity)
        body_up_world = self.state.quaternion.to_rotation_matrix()[:, 2]
        tilt_angle = np.arccos(np.clip(body_up_world[2], -1.0, 1.0))
        heading_error = abs(self._heading_error())
        return bool(
            position_error <= self.success_config.position_error
            and speed <= self.success_config.speed
            and angular_rate <= self.success_config.angular_rate
            and tilt_angle <= self.success_config.tilt_angle
            and heading_error <= self.success_config.heading_error
        )

    def is_terminated(self) -> bool:
        """Return whether a physical failure bound has been reached."""
        position_error = self._target_position - self.state.position
        body_up_world = self.state.quaternion.to_rotation_matrix()[:, 2]
        # Position and tilt bounds define states outside the recoverable task region.
        # High speed and angular rate remain recoverable and are handled by the reward.
        return bool(
            np.linalg.norm(position_error) >= self.max_position_error
            or body_up_world[2] <= np.cos(self.max_tilt_angle)
        )

    def is_truncated(self) -> bool:
        """Return whether the fixed episode time limit has been reached."""
        return self.episode_time >= self.episode_horizon

    def _nominal_state(self) -> QuadrotorState:
        half_heading = self._target_heading / 2.0
        return QuadrotorState(
            position=self._target_position,
            velocity=np.zeros(3),
            quaternion=Quaternion(
                np.cos(half_heading),
                0.0,
                0.0,
                np.sin(half_heading),
            ),
            angular_velocity=np.zeros(3),
        )

    def _set_hover_thrust(self) -> None:
        hover_thrust = self.params.mass * self.params.gravity / QDIMS.motor_count
        self.quadrotor.set_thrusts(
            np.full(QDIMS.motor_count, hover_thrust)
        )

    def _heading_error(self) -> float:
        error = self._target_heading - self.state.quaternion.to_yaw()
        return float(np.arctan2(np.sin(error), np.cos(error)))

    def reset(self, initial_state: QuadrotorState | None = None) -> QuadrotorState:
        """Reset episode time and optionally use a supplied initial state."""
        if initial_state is None:
            initial_state = self.initializer.initialize()
        elif not isinstance(initial_state, QuadrotorState):
            raise TypeError("initial_state must be a QuadrotorState")
        self.quadrotor.state = QuadrotorState(
            position=initial_state.position,
            velocity=initial_state.velocity,
            quaternion=initial_state.quaternion,
            angular_velocity=initial_state.angular_velocity,
        )
        self._set_hover_thrust()
        self.episode_time = 0.0
        self._last_raw_cost = 0.0
        self._last_normalized_cost = 0.0
        return self.quadrotor.state
