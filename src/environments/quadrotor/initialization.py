"""Initial-state samplers for the quadrotor environments."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from common.types import FloatVector
from experiments.config import RandomInitializerConfig
from geometry.rotation import Quaternion
from robots.quadrotor import QuadrotorState


class RandomInitializer:
    def __init__(
        self,
        config: RandomInitializerConfig,
        target_position: FloatVector,
        seed: int,
        *,
        target_heading: float = 0.0,
    ) -> None:
        if not isinstance(config, RandomInitializerConfig):
            raise TypeError("config must be RandomInitializerConfig")
        if config.name != "random":
            raise ValueError("RandomInitializer requires initializer name 'random'")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        target = np.array(target_position, dtype=np.float64, copy=True)
        if target.shape != (3,):
            raise ValueError(
                "target_position must have shape (3,), "
                f"got {target.shape}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_position must contain only finite values")
        if not np.isfinite(target_heading):
            raise ValueError("target_heading must be finite")
        self.config = config
        self._target_position = target
        self._target_heading = float(target_heading)
        self._rng = np.random.default_rng(int(seed))

    def initialize(self) -> QuadrotorState:
        # Position is sampled in a spherical shell around the hover target.
        # Speeds use random directions; tilt uses a random horizontal axis.
        return QuadrotorState(
            position=self._target_position + self._sample_nonnegative_vector(
                self.config.min_position_distance,
                self.config.max_position_distance,
            ),
            velocity=self._sample_nonnegative_vector(
                self.config.min_velocity,
                self.config.max_velocity,
            ),
            quaternion=self._sample_tilt(),
            angular_velocity=self._sample_nonnegative_vector(
                self.config.min_angular_velocity,
                self.config.max_angular_velocity,
            ),
        )

    def _sample_nonnegative_vector(
        self,
        minimum: float,
        maximum: float,
    ) -> FloatVector:
        if maximum == 0.0:
            return np.zeros(3)
        magnitude = self._rng.uniform(minimum, maximum)
        direction = self._rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        return magnitude * direction

    def _sample_tilt(self) -> Quaternion:
        if self.config.max_tilt_angle == 0.0:
            tilt = Quaternion.identity()
        else:
            angle = self._rng.uniform(
                self.config.min_tilt_angle,
                self.config.max_tilt_angle,
            )
            axis_heading = self._rng.uniform(0.0, 2.0 * np.pi)
            axis = np.array([np.cos(axis_heading), np.sin(axis_heading), 0.0])
            half_angle = angle / 2.0
            tilt = Quaternion(
                np.cos(half_angle),
                *(np.sin(half_angle) * axis),
            )
        heading = Quaternion(
            np.cos(self._target_heading / 2.0),
            0.0,
            0.0,
            np.sin(self._target_heading / 2.0),
        )
        return heading * tilt
