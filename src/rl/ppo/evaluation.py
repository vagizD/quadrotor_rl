"""Deterministic evaluation for the PPO hover policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from common.dimensions import QDIMS
from environments.quadrotor import HoverEnvironment, RandomInitializer
from experiments import EvaluationConfig
from geometry.rotation import Quaternion
from robots.quadrotor import QuadrotorState
from simulation import Trajectory

from .policy import Actor
from .normalization import ObservationNormalizer


@dataclass(frozen=True)
class EvaluationEpisode:
    trajectory: Trajectory
    rms_position_error: float
    final_position_error: float
    mean_speed: float
    maximum_angular_rate: float
    final_tilt_angle: float
    final_heading_error: float
    terminated: bool
    truncated: bool
    success: bool


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: tuple[EvaluationEpisode, ...]

    def as_metrics(self, update: int) -> dict[str, float | int]:
        if (
            isinstance(update, bool)
            or not isinstance(update, Integral)
            or update < 0
        ):
            raise ValueError("update must be a non-negative integer")
        return {
            "update": int(update),
            "evaluation_rms_position_error": float(
                np.mean([episode.rms_position_error for episode in self.episodes])
            ),
            "evaluation_final_position_error": float(
                np.mean([episode.final_position_error for episode in self.episodes])
            ),
            "evaluation_mean_speed": float(
                np.mean([episode.mean_speed for episode in self.episodes])
            ),
            "evaluation_max_angular_rate": float(
                np.max([episode.maximum_angular_rate for episode in self.episodes])
            ),
            "evaluation_success_rate": float(
                np.mean([episode.success for episode in self.episodes])
            ),
            "evaluation_failure_rate": float(
                np.mean([episode.terminated for episode in self.episodes])
            ),
            "evaluation_truncation_rate": float(
                np.mean([episode.truncated for episode in self.episodes])
            ),
        }


class DeterministicEvaluator:
    def __init__(
        self,
        environment: HoverEnvironment,
        actor: Actor,
        config: EvaluationConfig,
        initial_states: Sequence[QuadrotorState] | None = None,
        observation_normalizer: ObservationNormalizer | None = None,
    ) -> None:
        if not isinstance(environment, HoverEnvironment):
            raise TypeError("environment must be a HoverEnvironment")
        if not isinstance(actor, Actor):
            raise TypeError("actor must be an Actor")
        if not isinstance(config, EvaluationConfig):
            raise TypeError("config must be an EvaluationConfig")
        if actor.observation_dim != environment.observation_dim:
            raise ValueError("actor observation_dim must match environment")
        if actor.action_dim != environment.action_dim:
            raise ValueError("actor action_dim must match environment")
        if observation_normalizer is not None and (
            observation_normalizer.observation_dim != environment.observation_dim
        ):
            raise ValueError(
                "observation_normalizer dimension must match environment"
            )

        if initial_states is None:
            initializer = RandomInitializer(
                environment.initializer.config,
                environment.target_position,
                config.seed,
                target_heading=environment.target_heading,
            )
            states = tuple(
                initializer.initialize() for _ in range(config.episodes)
            )
        else:
            states = tuple(initial_states)
            if len(states) != config.episodes:
                raise ValueError("initial_states length must match evaluation episodes")
            if any(not isinstance(state, QuadrotorState) for state in states):
                raise TypeError("initial_states must contain QuadrotorState objects")
        self.environment = environment
        self.actor = actor
        self.config = config
        self.observation_normalizer = observation_normalizer
        self._initial_states = tuple(self._copy_state(state) for state in states)

    @property
    def initial_states(self) -> tuple[QuadrotorState, ...]:
        return tuple(self._copy_state(state) for state in self._initial_states)

    def should_evaluate(self, update: int) -> bool:
        if (
            isinstance(update, bool)
            or not isinstance(update, Integral)
            or update < 0
        ):
            raise ValueError("update must be a non-negative integer")
        return update > 0 and update % self.config.interval_updates == 0

    def evaluate(self) -> EvaluationSummary:
        was_training = self.actor.training
        self.actor.eval()
        episodes: list[EvaluationEpisode] = []
        try:
            for initial_state in self._initial_states:
                trajectory, terminated, truncated = self._run_episode(initial_state)
                episodes.append(
                    self._measure_episode(trajectory, terminated, truncated)
                )
        finally:
            self.actor.train(was_training)
        return EvaluationSummary(tuple(episodes))

    def _run_episode(
        self,
        initial_state: QuadrotorState,
    ) -> tuple[Trajectory, bool, bool]:
        self.environment.reset(initial_state)
        raw_observation = self.environment.get_observation()
        observation = self._normalize_observation(raw_observation)
        times = [0.0]
        states = [self.environment.state.to_vector()]
        thrusts: list[np.ndarray] = []
        max_steps = int(
            np.ceil(
                self.environment.episode_horizon / self.environment.integrator.dt
            )
        ) + 1
        parameter = next(self.actor.parameters())

        for step_index in range(max_steps):
            observation_tensor = torch.as_tensor(
                observation,
                dtype=parameter.dtype,
                device=parameter.device,
            )
            with torch.no_grad():
                # The distribution mean is the deterministic policy action;
                # no exploration sample is drawn during evaluation.
                action = self.actor.distribution(
                    observation_tensor,
                    action_low=0.0,
                    action_high=self.environment.params.max_thrust,
                ).mean
            action_array = action.detach().cpu().numpy()
            next_raw_observation, _, terminated, truncated = self.environment.step(
                action_array
            )
            thrusts.append(np.array(action_array, dtype=np.float64, copy=True))
            states.append(self.environment.state.to_vector())
            times.append((step_index + 1) * self.environment.integrator.dt)
            if terminated or truncated:
                break
            raw_observation = next_raw_observation
            observation = self._normalize_observation(raw_observation)
        else:
            raise RuntimeError("evaluation episode exceeded its configured horizon")

        return (
            Trajectory(
                times=np.asarray(times, dtype=np.float64),
                states=np.asarray(states, dtype=np.float64),
                thrusts=np.asarray(thrusts, dtype=np.float64),
                target_positions=self.environment.target_position,
            ),
            bool(terminated),
            bool(truncated),
        )

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.observation_normalizer is None:
            return observation
        return self.observation_normalizer.normalize(observation)

    def _measure_episode(
        self,
        trajectory: Trajectory,
        terminated: bool,
        truncated: bool,
    ) -> EvaluationEpisode:
        target = trajectory.target_positions
        if target is None or target.shape != (3,):
            raise RuntimeError("evaluation trajectory must have a fixed target")
        position_error = np.linalg.norm(
            trajectory.states[:, QDIMS.position_slice] - target,
            axis=1,
        )
        speed = np.linalg.norm(
            trajectory.states[:, QDIMS.velocity_slice],
            axis=1,
        )
        angular_rate = np.linalg.norm(
            trajectory.states[:, QDIMS.angular_velocity_slice],
            axis=1,
        )
        final_state = QuadrotorState.from_vector(trajectory.states[-1])
        body_up_world = final_state.quaternion.to_rotation_matrix()[:, 2]
        final_tilt_angle = float(
            np.arccos(np.clip(body_up_world[2], -1.0, 1.0))
        )
        heading_error = self.environment.target_heading - final_state.quaternion.to_yaw()
        final_heading_error = float(
            abs(np.arctan2(np.sin(heading_error), np.cos(heading_error)))
        )
        final_position_error = float(position_error[-1])
        # The environment owns the single success definition used by reward and evaluation.
        success = bool(not terminated and self.environment.is_successful())
        return EvaluationEpisode(
            trajectory=trajectory,
            rms_position_error=float(np.sqrt(np.mean(position_error**2))),
            final_position_error=final_position_error,
            mean_speed=float(np.mean(speed)),
            maximum_angular_rate=float(np.max(angular_rate)),
            final_tilt_angle=final_tilt_angle,
            final_heading_error=final_heading_error,
            terminated=terminated,
            truncated=truncated,
            success=success,
        )

    @staticmethod
    def _copy_state(state: QuadrotorState) -> QuadrotorState:
        return QuadrotorState(
            position=state.position,
            velocity=state.velocity,
            quaternion=Quaternion(
                state.quaternion.w,
                state.quaternion.x,
                state.quaternion.y,
                state.quaternion.z,
                normalize=False,
            ),
            angular_velocity=state.angular_velocity,
        )
