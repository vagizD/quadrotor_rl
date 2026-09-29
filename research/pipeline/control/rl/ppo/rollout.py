"""PPO trajectory segments, rollout storage, and collection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch

from geometry.common.types import FloatMatrix, FloatVector
from research.evaluation.quadrotor.scenarios import HoverEnvironment
from research.pipeline.planning.trajectory import Trajectory

from .policy import Actor, Critic
from .normalization import ObservationNormalizer


def _copy_finite_array(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _copy_flags(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.bool_:
        raise TypeError(f"{name} must have Boolean dtype")
    return np.array(value, dtype=bool, copy=True)


def _require_unit_interval(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


class PPOTrajectory(Trajectory):
    """Trajectory with PPO transition data."""

    def __init__(
        self,
        times: FloatVector,
        states: FloatMatrix,
        thrusts: FloatMatrix,
        observations: FloatMatrix,
        rewards: FloatVector,
        values: FloatVector,
        log_probabilities: FloatVector,
        terminated: np.ndarray,
        truncated: np.ndarray,
        target_positions: FloatVector | FloatMatrix | None = None,
        boundary_state_value: float = 0.0,
        returns: FloatVector | None = None,
        advantages: FloatVector | None = None,
        raw_observations: FloatMatrix | None = None,
        raw_costs: FloatVector | None = None,
        normalized_costs: FloatVector | None = None,
    ) -> None:
        super().__init__(times, states, thrusts, target_positions)
        step_count = self.step_count
        self.observations = _copy_finite_array(observations, "observations")
        self.raw_observations = (
            self.observations.copy()
            if raw_observations is None
            else _copy_finite_array(raw_observations, "raw_observations")
        )
        self.rewards = _copy_finite_array(rewards, "rewards")
        self.values = _copy_finite_array(values, "values")
        self.log_probabilities = _copy_finite_array(
            log_probabilities,
            "log_probabilities",
        )
        self.terminated = _copy_flags(terminated, "terminated")
        self.truncated = _copy_flags(truncated, "truncated")
        if not np.isfinite(boundary_state_value):
            raise ValueError("boundary_state_value must be finite")
        self.boundary_state_value = float(boundary_state_value)

        if self.observations.ndim != 2 or self.observations.shape[0] != step_count:
            raise ValueError(
                "observations must have shape (N, observation_dim), "
                f"got {self.observations.shape}"
            )
        if (
            self.raw_observations.ndim != 2
            or self.raw_observations.shape != self.observations.shape
        ):
            raise ValueError(
                "raw_observations must have the same shape as observations, "
                f"got {self.raw_observations.shape}"
            )
        for name, array in (
            ("rewards", self.rewards),
            ("values", self.values),
            ("log_probabilities", self.log_probabilities),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
        ):
            if array.shape != (step_count,):
                raise ValueError(
                    f"{name} must have shape (N,), got {array.shape}"
                )
        self.returns = self._copy_optional_vector(returns, "returns", step_count)
        self.advantages = self._copy_optional_vector(
            advantages,
            "advantages",
            step_count,
        )
        self.raw_costs = self._copy_optional_vector(
            raw_costs,
            "raw_costs",
            step_count,
        )
        self.normalized_costs = self._copy_optional_vector(
            normalized_costs,
            "normalized_costs",
            step_count,
        )

    @property
    def actions(self) -> FloatMatrix:
        return self.thrusts

    @staticmethod
    def _copy_optional_vector(
        value: FloatVector | None,
        name: str,
        step_count: int,
    ) -> FloatVector | None:
        if value is None:
            return None
        vector = _copy_finite_array(value, name)
        if vector.shape != (step_count,):
            raise ValueError(f"{name} must have shape (N,), got {vector.shape}")
        return vector

    def compute_returns(
        self,
        gamma: float,
        reward_scale: float = 1.0,
    ) -> FloatVector:
        """Compute discounted return-to-go values."""
        gamma = _require_unit_interval(gamma, "gamma")
        reward_scale = _require_positive(reward_scale, "reward_scale")
        returns = np.empty(self.step_count, dtype=np.float64)
        future_return = self.boundary_state_value
        for index in range(self.step_count - 1, -1, -1):
            # A physical terminal state has no future value. A time-limit
            # truncation remains bootstrap-able, so it is not used as a mask.
            continuation = 0.0 if self.terminated[index] else 1.0
            future_return = (
                reward_scale * self.rewards[index]
                + gamma * continuation * future_return
            )
            returns[index] = future_return
        self.returns = returns
        return self.returns

    def compute_gae(
        self,
        gamma: float,
        gae_lambda: float,
        reward_scale: float = 1.0,
    ) -> FloatVector:
        """Compute generalized advantage estimates."""
        gamma = _require_unit_interval(gamma, "gamma")
        gae_lambda = _require_unit_interval(gae_lambda, "gae_lambda")
        reward_scale = _require_positive(reward_scale, "reward_scale")
        advantages = np.empty(self.step_count, dtype=np.float64)
        next_value = self.boundary_state_value
        next_advantage = 0.0
        for index in range(self.step_count - 1, -1, -1):
            # The value at the next state is valid after truncation but not
            # after physical termination.
            continuation = 0.0 if self.terminated[index] else 1.0
            temporal_difference = (
                reward_scale * self.rewards[index]
                + gamma * continuation * next_value
                - self.values[index]
            )
            next_advantage = temporal_difference + (
                gamma * gae_lambda * continuation * next_advantage
            )
            advantages[index] = next_advantage
            next_value = self.values[index]
        self.advantages = advantages
        return self.advantages

    def _extra_arrays_for_save(self) -> dict[str, np.ndarray]:
        arrays = {
            "observations": self.observations,
            "raw_observations": self.raw_observations,
            "rewards": self.rewards,
            "values": self.values,
            "log_probabilities": self.log_probabilities,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "boundary_state_value": np.array(self.boundary_state_value),
            "returns": (
                np.empty(0, dtype=np.float64)
                if self.returns is None
                else self.returns
            ),
            "has_returns": np.array(self.returns is not None),
            "advantages": (
                np.empty(0, dtype=np.float64)
                if self.advantages is None
                else self.advantages
            ),
            "has_advantages": np.array(self.advantages is not None),
            "raw_costs": (
                np.empty(0, dtype=np.float64)
                if self.raw_costs is None
                else self.raw_costs
            ),
            "has_raw_costs": np.array(self.raw_costs is not None),
            "normalized_costs": (
                np.empty(0, dtype=np.float64)
                if self.normalized_costs is None
                else self.normalized_costs
            ),
            "has_normalized_costs": np.array(self.normalized_costs is not None),
        }
        return arrays

    @classmethod
    def _load_extra_arrays(cls, data) -> dict[str, object]:
        names = (
            "observations",
            "rewards",
            "values",
            "log_probabilities",
            "terminated",
            "truncated",
            "raw_observations",
        )
        required_names = names[:-1]
        missing = [name for name in required_names if name not in data]
        if missing:
            raise ValueError(
                "PPO trajectory is missing arrays: " + ", ".join(missing)
            )
        arrays: dict[str, object] = {
            name: data[name]
            for name in required_names
        }
        if "raw_observations" not in data:
            arrays["raw_observations"] = arrays["observations"]
        arrays["boundary_state_value"] = (
            float(data["boundary_state_value"].item())
            if "boundary_state_value" in data
            else 0.0
        )
        arrays["returns"] = (
            data["returns"]
            if "has_returns" in data and bool(data["has_returns"].item())
            else None
        )
        arrays["advantages"] = (
            data["advantages"]
            if "has_advantages" in data
            and bool(data["has_advantages"].item())
            else None
        )
        arrays["raw_costs"] = (
            data["raw_costs"]
            if "has_raw_costs" in data and bool(data["has_raw_costs"].item())
            else None
        )
        arrays["normalized_costs"] = (
            data["normalized_costs"]
            if "has_normalized_costs" in data
            and bool(data["has_normalized_costs"].item())
            else None
        )
        return arrays


@dataclass(frozen=True)
class PPOBatch:
    observations: FloatMatrix
    actions: FloatMatrix
    rewards: FloatVector
    values: FloatVector
    log_probabilities: FloatVector
    returns: FloatVector
    advantages: FloatVector
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def step_count(self) -> int:
        return self.rewards.shape[0]


class PPORollout:
    """Collection of trajectory segments used for one PPO update."""

    def __init__(self, trajectories: Sequence[PPOTrajectory]) -> None:
        trajectories = tuple(trajectories)
        if not trajectories:
            raise ValueError("trajectories must not be empty")
        if any(
            not isinstance(trajectory, PPOTrajectory)
            for trajectory in trajectories
        ):
            raise TypeError("trajectories must contain PPOTrajectory objects")
        self.trajectories = trajectories

    @property
    def trajectory_count(self) -> int:
        return len(self.trajectories)

    @property
    def step_count(self) -> int:
        return sum(trajectory.step_count for trajectory in self.trajectories)

    def compute_targets(
        self,
        gamma: float,
        gae_lambda: float,
        reward_scale: float = 1.0,
    ) -> None:
        for trajectory in self.trajectories:
            trajectory.compute_returns(gamma, reward_scale)
            trajectory.compute_gae(gamma, gae_lambda, reward_scale)

    def to_batch(self) -> PPOBatch:
        if any(
            trajectory.returns is None or trajectory.advantages is None
            for trajectory in self.trajectories
        ):
            raise RuntimeError("compute returns and advantages before creating a batch")
        return PPOBatch(
            observations=np.concatenate(
                [trajectory.observations for trajectory in self.trajectories],
                axis=0,
            ),
            actions=np.concatenate(
                [trajectory.actions for trajectory in self.trajectories],
                axis=0,
            ),
            rewards=np.concatenate(
                [trajectory.rewards for trajectory in self.trajectories],
                axis=0,
            ),
            values=np.concatenate(
                [trajectory.values for trajectory in self.trajectories],
                axis=0,
            ),
            log_probabilities=np.concatenate(
                [
                    trajectory.log_probabilities
                    for trajectory in self.trajectories
                ],
                axis=0,
            ),
            returns=np.concatenate(
                [trajectory.returns for trajectory in self.trajectories],
                axis=0,
            ),
            advantages=np.concatenate(
                [trajectory.advantages for trajectory in self.trajectories],
                axis=0,
            ),
            terminated=np.concatenate(
                [trajectory.terminated for trajectory in self.trajectories],
                axis=0,
            ),
            truncated=np.concatenate(
                [trajectory.truncated for trajectory in self.trajectories],
                axis=0,
            ),
        )


def _validate_collection_inputs(
    environment: HoverEnvironment,
    actor: Actor,
    critic: Critic,
    steps: int,
) -> None:
    if not isinstance(environment, HoverEnvironment):
        raise TypeError("environment must be HoverEnvironment")
    if not isinstance(actor, Actor):
        raise TypeError("actor must be Actor")
    if not isinstance(critic, Critic):
        raise TypeError("critic must be Critic")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if actor.observation_dim != environment.observation_dim:
        raise ValueError("actor observation_dim must match environment")
    if actor.action_dim != environment.action_dim:
        raise ValueError("actor action_dim must match environment")
    if critic.observation_dim != environment.observation_dim:
        raise ValueError("critic observation_dim must match environment")


def collect_ppo_trajectory(
    environment: HoverEnvironment,
    actor: Actor,
    critic: Critic,
    steps: int,
    observation_normalizer: ObservationNormalizer | None = None,
    normalization_statistics: tuple[FloatVector, FloatVector] | None = None,
) -> PPOTrajectory:
    """Collect one contiguous PPO trajectory segment."""
    _validate_collection_inputs(environment, actor, critic, steps)
    if observation_normalizer is None and normalization_statistics is not None:
        raise ValueError(
            "normalization_statistics requires an observation_normalizer"
        )
    if observation_normalizer is not None:
        if observation_normalizer.observation_dim != environment.observation_dim:
            raise ValueError(
                "observation_normalizer dimension must match environment"
            )
        if normalization_statistics is None:
            normalization_statistics = observation_normalizer.snapshot()

    environment.reset()
    raw_observation = environment.get_observation()
    observation = (
        raw_observation
        if observation_normalizer is None
        else observation_normalizer.normalize(
            raw_observation,
            normalization_statistics,
        )
    )
    times = [0.0]
    states = [environment.state.to_vector()]
    observations: list[np.ndarray] = []
    raw_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    values: list[float] = []
    log_probabilities: list[float] = []
    raw_costs: list[float] = []
    normalized_costs: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []

    parameter = next(actor.parameters())
    for step_index in range(steps):
        observation_tensor = torch.as_tensor(
            observation,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        with torch.no_grad():
            distribution = actor.distribution(
                observation_tensor,
                action_low=0.0,
                action_high=environment.params.max_thrust,
            )
            action = distribution.sample()
            log_probability = distribution.log_prob(action)
            value = critic(observation_tensor)

        action_array = action.detach().cpu().numpy()
        next_observation, reward, is_terminated, is_truncated = environment.step(
            action_array
        )

        observations.append(np.array(observation, dtype=np.float64, copy=True))
        raw_observations.append(
            np.array(raw_observation, dtype=np.float64, copy=True)
        )
        actions.append(np.array(action_array, dtype=np.float64, copy=True))
        rewards.append(float(reward))
        raw_costs.append(environment.last_raw_cost)
        normalized_costs.append(environment.last_normalized_cost)
        values.append(float(value.item()))
        log_probabilities.append(float(log_probability.item()))
        terminated.append(bool(is_terminated))
        truncated.append(bool(is_truncated))
        states.append(environment.state.to_vector())
        times.append((step_index + 1) * environment.integrator.dt)

        if is_terminated or is_truncated:
            break
        raw_observation = next_observation
        observation = (
            raw_observation
            if observation_normalizer is None
            else observation_normalizer.normalize(
                raw_observation,
                normalization_statistics,
            )
        )

    boundary_state_value = 0.0
    if not terminated[-1]:
        critic_parameter = next(critic.parameters())
        next_observation_tensor = torch.as_tensor(
            next_observation,
            dtype=critic_parameter.dtype,
            device=critic_parameter.device,
        )
        with torch.no_grad():
            boundary_state_value = float(critic(next_observation_tensor).item())

    return PPOTrajectory(
        times=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        thrusts=np.asarray(actions, dtype=np.float64),
        observations=np.asarray(observations, dtype=np.float64),
        rewards=np.asarray(rewards, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        log_probabilities=np.asarray(log_probabilities, dtype=np.float64),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
        target_positions=environment.target_position,
        boundary_state_value=boundary_state_value,
        raw_observations=np.asarray(raw_observations, dtype=np.float64),
        raw_costs=np.asarray(raw_costs, dtype=np.float64),
        normalized_costs=np.asarray(normalized_costs, dtype=np.float64),
    )


def collect_ppo_rollout(
    environment: HoverEnvironment,
    actor: Actor,
    critic: Critic,
    steps: int,
    observation_normalizer: ObservationNormalizer | None = None,
) -> PPORollout:
    """Collect segments until the rollout reaches the requested step count."""
    _validate_collection_inputs(environment, actor, critic, steps)
    normalization_statistics = (
        None
        if observation_normalizer is None
        else observation_normalizer.snapshot()
    )
    # Keep one normalization snapshot for all segments in this on-policy batch.
    trajectories: list[PPOTrajectory] = []
    remaining_steps = steps
    while remaining_steps > 0:
        trajectory = collect_ppo_trajectory(
            environment,
            actor,
            critic,
            remaining_steps,
            observation_normalizer=observation_normalizer,
            normalization_statistics=normalization_statistics,
        )
        trajectories.append(trajectory)
        remaining_steps -= trajectory.step_count
    if observation_normalizer is not None:
        # Update only after collection so stored policy log-probabilities remain aligned.
        observation_normalizer.update(
            np.concatenate(
                [trajectory.raw_observations for trajectory in trajectories],
                axis=0,
            )
        )
    return PPORollout(trajectories)


def _require_positive(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)
