"""Quadrotor state container."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from common.types import FloatVector
from geometry.rotation import Quaternion


def _as_vector(value: ArrayLike, name: str, size: int) -> FloatVector:
    """Convert a state component to a validated float vector."""
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    return vector


@dataclass
class QuadrotorState:
    """State components for one quadrotor."""

    position: FloatVector
    velocity: FloatVector
    quaternion: Quaternion
    angular_velocity: FloatVector

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, "position", 3)
        self.velocity = _as_vector(self.velocity, "velocity", 3)
        if not isinstance(self.quaternion, Quaternion):
            raise TypeError("quaternion must be a Quaternion")
        if not np.isclose(self.quaternion.norm(), 1.0):
            raise ValueError("quaternion must have unit norm")
        self.angular_velocity = _as_vector(
            self.angular_velocity,
            "angular_velocity",
            3,
        )

    def to_vector(self) -> FloatVector:
        return np.concatenate(
            [
                self.position,
                self.velocity,
                self.quaternion.as_array(),
                self.angular_velocity,
            ]
        )

    @classmethod
    def from_vector(cls, value: FloatVector) -> "QuadrotorState":
        vector = _as_vector(value, "state", 13)
        return cls(
            position=vector[0:3],
            velocity=vector[3:6],
            quaternion=Quaternion.from_array(vector[6:10]),
            angular_velocity=vector[10:13],
        )


@dataclass
class QuadrotorStateDerivative:
    """Time derivative of a quadrotor state."""

    position: FloatVector
    velocity: FloatVector
    quaternion: FloatVector
    angular_velocity: FloatVector

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, "position derivative", 3)
        self.velocity = _as_vector(self.velocity, "velocity derivative", 3)
        self.quaternion = _as_vector(self.quaternion, "quaternion derivative", 4)
        self.angular_velocity = _as_vector(
            self.angular_velocity,
            "angular velocity derivative",
            3,
        )

    def to_vector(self) -> FloatVector:
        return np.concatenate(
            [
                self.position,
                self.velocity,
                self.quaternion,
                self.angular_velocity,
            ]
        )
