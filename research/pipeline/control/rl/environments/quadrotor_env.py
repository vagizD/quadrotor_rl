"""Quadrotor hover task state and reset behavior."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from geometry.common.dimensions import QDIMS
from geometry.common.types import FloatVector
from research.pipeline.control.residual.models.experiments import (
    DimensionsConfig,
    ExperimentConfig,
    HoverEnvironmentConfig,
    RewardConfig,
)
from geometry.math.rotation import Quaternion
from .initialization import RandomInitializer
from robots.quadrotor import (
    Quadrotor,
    QuadrotorParams,
    RigidBodyState,
    compute_state_derivative_vector,
)
from research.simulator.simulator import QuadrotorEnvironment

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
    "hover_skill_gate",
)

class QuadrotorRLEnvironment(QuadrotorEnvironment):
    """Reinforcement learning wrapper for the quadrotor physical simulator."""

    def __init__(
        self,
        params: QuadrotorParams,
        integrator: EulerIntegrator | RK4Integrator,
        dimensions: DimensionsConfig,
        environment_config: HoverEnvironmentConfig,
        reward_config: RewardConfig,
        initializer: RandomInitializer,
    ) -> None:
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

        # Nominal initialization state
        half_heading = environment_config.target_heading / 2.0
        initial_state = RigidBodyState(
            position=environment_config.target_position.copy(),
            velocity=np.zeros(3),
            quaternion=Quaternion(
                np.cos(half_heading),
                0.0,
                0.0,
                np.sin(half_heading),
            ),
            angular_velocity=np.zeros(3),
        )

        super().__init__(
            params=params,
            integrator=integrator,
            initial_state=initial_state,
            disturbances=[],  # Will be populated via config
        )

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
        
        self._last_raw_cost = 0.0
        self._last_normalized_cost = 0.0
        self._set_hover_thrust()

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "QuadrotorRLEnvironment":
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
    def last_raw_cost(self) -> float:
        return self._last_raw_cost

    @property
    def last_normalized_cost(self) -> float:
        return self._last_normalized_cost

    def get_observation(self) -> FloatVector:
        pos_err = self._target_position - self.state.position
        dist = float(np.linalg.norm(pos_err))
        # Non-linear Gaussian skill gate: 1.0 at target, 0.0 far away
        hover_gate = np.exp(-(dist**2) / (2.0 * (0.30**2)))
        observation = np.concatenate(
            [
                pos_err,
                self.state.velocity,
                self.state.quaternion.as_array(),
                self.state.angular_velocity,
                np.array([hover_gate], dtype=np.float64),
            ]
        )
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                "hover observation shape does not match configured observation_dim"
            )
        return observation

    def step(self, action: ArrayLike) -> tuple[FloatVector, float, bool, bool]:
        """Apply one action and return observation, reward, and episode flags."""
        super().step(action)
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

        # Skill-Gated Task-Conditioned Dual-Mode Reward:
        hover_gate = float(
            np.exp(
                -(distance_to_target**2)
                / (2.0 * (self.reward_config.hover_gate_distance_scale**2))
            )
        )

        # Transit cost: linear distance penalty + soft allowed-tilt window penalty (zero penalty for tilt <= 30 deg)
        tilt_excess = max(
            0.0,
            self.reward_config.transit_allowed_tilt_z - body_up_world[2],
        )
        transit_cost = (
            self.reward_config.position_error_weight * distance_to_target
            + self.reward_config.linear_velocity_weight
            * float(np.dot(self.state.velocity, self.state.velocity))
            + self.reward_config.tilt_weight * (tilt_excess**2)
            + self.reward_config.heading_weight * heading_error**2
        )

        mean_thrust = np.mean(self.quadrotor.thrusts)
        thrust_asymmetry = self.quadrotor.thrusts - mean_thrust
        thrust_asymmetry_cost = (
            self.reward_config.hover_thrust_asymmetry_weight
            * np.dot(thrust_asymmetry, thrust_asymmetry)
        )

        hover_cost = (
            self.reward_config.position_error_weight * (distance_to_target**2)
            + self.reward_config.heading_weight * heading_error**2
            + (
                self.reward_config.linear_velocity_weight
                * self.reward_config.hover_velocity_multiplier
            )
            * np.dot(self.state.velocity, self.state.velocity)
            + (
                self.reward_config.angular_velocity_weight
                * self.reward_config.hover_angular_velocity_multiplier
            )
            * np.dot(
                self.state.angular_velocity,
                self.state.angular_velocity,
            )
            + self.reward_config.thrust_deviation_weight
            * np.dot(thrust_deviation, thrust_deviation)
            + thrust_asymmetry_cost
        )

        raw_cost = (1.0 - hover_gate) * transit_cost + hover_gate * hover_cost

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

    def _nominal_state(self) -> RigidBodyState:
        half_heading = self._target_heading / 2.0
        return RigidBodyState(
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

    def reset(self, initial_state: RigidBodyState | None = None) -> RigidBodyState:
        """Reset episode time and optionally use a supplied initial state."""
        if initial_state is None:
            initial_state = self.initializer.initialize()
        elif not isinstance(initial_state, RigidBodyState):
            raise TypeError("initial_state must be a RigidBodyState")
        
        super().reset(initial_state)
        
        self._set_hover_thrust()
        self._last_raw_cost = 0.0
        self._last_normalized_cost = 0.0
        return self.quadrotor.state
