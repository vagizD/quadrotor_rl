"""Running observation statistics for PPO inputs."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from geometry.common.types import FloatMatrix, FloatVector


class ObservationNormalizer:
    def __init__(
        self,
        observation_dim: int,
        clip_value: float,
        epsilon: float,
    ) -> None:
        if (
            isinstance(observation_dim, bool)
            or not isinstance(observation_dim, Integral)
            or observation_dim <= 0
        ):
            raise ValueError("observation_dim must be a positive integer")
        for value, name in (
            (clip_value, "clip_value"),
            (epsilon, "epsilon"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.observation_dim = int(observation_dim)
        self.clip_value = float(clip_value)
        self.epsilon = float(epsilon)
        self.mean = np.zeros(self.observation_dim, dtype=np.float64)
        self.variance = np.ones(self.observation_dim, dtype=np.float64)
        self.count = 0.0

    def snapshot(self) -> tuple[FloatVector, FloatVector]:
        return self.mean.copy(), self.variance.copy()

    def normalize(
        self,
        value: FloatVector | FloatMatrix,
        statistics: tuple[FloatVector, FloatVector] | None = None,
    ) -> np.ndarray:
        array = self._validate(value, "value")
        mean, variance = self.snapshot() if statistics is None else statistics
        normalized = (array - mean) / np.sqrt(variance + self.epsilon)
        return np.clip(normalized, -self.clip_value, self.clip_value)

    def update(self, values: FloatVector | FloatMatrix) -> None:
        array = self._validate(values, "values")
        batch = array[None, :] if array.ndim == 1 else array
        batch_count = float(batch.shape[0])
        batch_mean = np.mean(batch, axis=0)
        batch_variance = np.var(batch, axis=0)
        if self.count == 0.0:
            self.mean = batch_mean
            self.variance = batch_variance
            self.count = batch_count
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        first_moment = self.variance * self.count
        second_moment = batch_variance * batch_count
        self.mean += delta * batch_count / total_count
        self.variance = (
            first_moment
            + second_moment
            + np.square(delta) * self.count * batch_count / total_count
        ) / total_count
        self.count = total_count

    def state_dict(self) -> dict[str, object]:
        return {
            "observation_dim": self.observation_dim,
            "clip_value": self.clip_value,
            "epsilon": self.epsilon,
            "mean": self.mean.copy(),
            "variance": self.variance.copy(),
            "count": self.count,
        }

    def _validate(
        self,
        value: FloatVector | FloatMatrix,
        name: str,
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[-1] != self.observation_dim:
            raise ValueError(
                f"{name} must have shape (D,) or (N, D), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array
