"""Rigid body state container."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from geometry.common.dimensions import QDIMS
from geometry.common.types import FloatVector
from geometry.math.rotation import Quaternion

def _as_vector(value: ArrayLike, name: str, size: int) -> FloatVector:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    return vector

@dataclass
class RigidBodyState:
    position: FloatVector
    velocity: FloatVector
    quaternion: Quaternion
    angular_velocity: FloatVector

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, "position", QDIMS.position_dim)
        self.velocity = _as_vector(self.velocity, "velocity", QDIMS.velocity_dim)
        if not isinstance(self.quaternion, Quaternion):
            raise TypeError("quaternion must be a Quaternion")
        if not np.isclose(self.quaternion.norm(), 1.0):
            raise ValueError("quaternion must have unit norm")
        self.angular_velocity = _as_vector(self.angular_velocity, "angular_velocity", QDIMS.angular_velocity_dim)

    def to_vector(self) -> FloatVector:
        return np.concatenate([self.position, self.velocity, self.quaternion.as_array(), self.angular_velocity])

    @classmethod
    def from_vector(cls, value: FloatVector) -> "RigidBodyState":
        vector = _as_vector(value, "state", QDIMS.state_dim)
        return cls(
            position=vector[QDIMS.position_slice],
            velocity=vector[QDIMS.velocity_slice],
            quaternion=Quaternion.from_array(vector[QDIMS.quaternion_slice]),
            angular_velocity=vector[QDIMS.angular_velocity_slice],
        )

@dataclass
class RigidBodyStateDerivative:
    position: FloatVector
    velocity: FloatVector
    quaternion: FloatVector
    angular_velocity: FloatVector

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, "position derivative", QDIMS.position_dim)
        self.velocity = _as_vector(self.velocity, "velocity derivative", QDIMS.velocity_dim)
        self.quaternion = _as_vector(self.quaternion, "quaternion derivative", QDIMS.quaternion_dim)
        self.angular_velocity = _as_vector(self.angular_velocity, "angular velocity derivative", QDIMS.angular_velocity_dim)

    def to_vector(self) -> FloatVector:
        return np.concatenate([self.position, self.velocity, self.quaternion, self.angular_velocity])
