"""Recorded simulation trajectories."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geometry.common.dimensions import QDIMS
from geometry.common.types import FloatMatrix, FloatVector


def _copy_finite_array(value: FloatVector | FloatMatrix, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass
class Trajectory:
    """State and action samples recorded over simulation time."""

    times: FloatVector
    states: FloatMatrix
    thrusts: FloatMatrix
    target_positions: FloatVector | FloatMatrix | None = None
    disturbance_forces: FloatMatrix | None = None
    disturbance_torques: FloatMatrix | None = None

    disturbance_scale: float | None = None
    wind_vector: FloatVector | None = None
    wind_label: str | None = None

    def __post_init__(self) -> None:
        self.times = _copy_finite_array(self.times, "times")
        self.states = _copy_finite_array(self.states, "states")
        self.thrusts = _copy_finite_array(self.thrusts, "thrusts")

        if self.times.ndim != 1 or self.times.shape[0] == 0:
            raise ValueError("times must have shape (N + 1,) with N >= 0")
        if self.states.shape != (self.times.shape[0], QDIMS.state_dim):
            raise ValueError(
                "states must have shape "
                f"(N + 1, {QDIMS.state_dim}), "
                f"got {self.states.shape}"
            )
        if self.thrusts.shape != (
            self.times.shape[0] - 1,
            QDIMS.motor_count,
        ):
            raise ValueError(
                "thrusts must have shape "
                f"(N, {QDIMS.motor_count}), "
                f"got {self.thrusts.shape}"
            )
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if np.any(self.thrusts < 0.0):
            raise ValueError("thrusts must be non-negative")

        if self.target_positions is not None:
            targets = _copy_finite_array(self.target_positions, "target_positions")
            if targets.shape not in ((3,), (self.times.shape[0], 3)):
                raise ValueError(
                    "target_positions must have shape (3,) or (N + 1, 3), "
                    f"got {targets.shape}"
                )
            self.target_positions = targets
            
        if self.disturbance_forces is not None:
            forces = _copy_finite_array(self.disturbance_forces, "disturbance_forces")
            if forces.shape != (self.times.shape[0] - 1, 3):
                raise ValueError(f"disturbance_forces must have shape (N, 3)")
            self.disturbance_forces = forces
            
        if self.disturbance_torques is not None:
            torques = _copy_finite_array(self.disturbance_torques, "disturbance_torques")
            if torques.shape != (self.times.shape[0] - 1, 3):
                raise ValueError(f"disturbance_torques must have shape (N, 3)")
            self.disturbance_torques = torques

    @property
    def step_count(self) -> int:
        return self.times.shape[0] - 1

    def _extra_arrays_for_save(self) -> dict[str, np.ndarray]:
        extra = {}
        if self.disturbance_forces is not None:
            extra["disturbance_forces"] = self.disturbance_forces
        if self.disturbance_torques is not None:
            extra["disturbance_torques"] = self.disturbance_torques
        if self.disturbance_scale is not None:
            extra["disturbance_scale"] = np.array([self.disturbance_scale])
        if self.wind_vector is not None:
            extra["wind_vector"] = self.wind_vector
        if self.wind_label is not None:
            extra["wind_label"] = np.array([self.wind_label])
        return extra

    @classmethod
    def _load_extra_arrays(cls, data) -> dict[str, object]:
        extra = {}
        if "disturbance_forces" in data:
            extra["disturbance_forces"] = data["disturbance_forces"]
        if "disturbance_torques" in data:
            extra["disturbance_torques"] = data["disturbance_torques"]
        if "disturbance_scale" in data:
            extra["disturbance_scale"] = float(data["disturbance_scale"][0])
        if "wind_vector" in data:
            extra["wind_vector"] = data["wind_vector"]
        if "wind_label" in data:
            extra["wind_label"] = str(data["wind_label"][0])
        return extra

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target_positions = (
            np.empty(0, dtype=np.float64)
            if self.target_positions is None
            else self.target_positions
        )
        arrays = {
            "times": self.times,
            "states": self.states,
            "thrusts": self.thrusts,
            "target_positions": target_positions,
            "has_target": np.array(self.target_positions is not None),
        }
        arrays.update(self._extra_arrays_for_save())
        np.savez_compressed(output_path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        with np.load(path, allow_pickle=False) as data:
            target_positions = (
                data["target_positions"]
                if bool(data["has_target"].item())
                else None
            )
            extra_arrays = cls._load_extra_arrays(data)
            return cls(
                times=data["times"],
                states=data["states"],
                thrusts=data["thrusts"],
                target_positions=target_positions,
                **extra_arrays,
            )
