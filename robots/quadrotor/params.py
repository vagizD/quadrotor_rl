"""Quadrotor physical parameters."""

from dataclasses import dataclass, field

import numpy as np

from geometry.common.dimensions import QDIMS
from geometry.common.types import FloatMatrix, FloatVector


DEFAULT_MOTOR_SPIN_DIRECTIONS = np.array(
    [1.0, -1.0, 1.0, -1.0],
    dtype=np.float64,
)


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


@dataclass
class QuadrotorParams:
    """Validated physical parameters for one quadrotor."""

    mass: float
    arm_length: float
    inertia: FloatMatrix
    yaw_torque_coefficient: float
    max_thrust: float
    gravity: float = 9.81
    motor_spin_directions: FloatVector = field(
        default_factory=lambda: DEFAULT_MOTOR_SPIN_DIRECTIONS.copy(),
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("mass", self.mass),
            ("arm_length", self.arm_length),
            ("yaw_torque_coefficient", self.yaw_torque_coefficient),
            ("max_thrust", self.max_thrust),
            ("gravity", self.gravity),
        ):
            _require_positive(name, value)

        inertia = np.array(self.inertia, dtype=np.float64, copy=True)
        if inertia.shape != (3, 3):
            raise ValueError(f"inertia must have shape (3, 3), got {inertia.shape}")
        if not np.all(np.isfinite(inertia)):
            raise ValueError("inertia must contain only finite values")
        if not np.allclose(inertia, np.diag(np.diag(inertia))):
            raise ValueError("inertia must be diagonal")
        if np.any(np.diag(inertia) <= 0.0):
            raise ValueError("inertia diagonal must be positive")
        self.inertia = inertia

        directions = np.array(
            self.motor_spin_directions,
            dtype=np.float64,
            copy=True,
        )
        if directions.shape != (QDIMS.motor_count,):
            raise ValueError(
                "motor_spin_directions must have shape "
                f"({QDIMS.motor_count},), "
                f"got {directions.shape}"
            )
        if not np.all(np.isin(directions, (-1.0, 1.0))):
            raise ValueError("motor_spin_directions must contain only -1 or 1")
        self.motor_spin_directions = directions
