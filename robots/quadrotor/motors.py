"""Quadrotor motor force and torque mapping."""

import numpy as np

from geometry.common.dimensions import QDIMS
from geometry.common.types import FloatVector


def _require_thrusts(thrusts: FloatVector) -> None:
    if not isinstance(thrusts, np.ndarray):
        raise TypeError("thrusts must be a NumPy array")
    if thrusts.shape != (QDIMS.motor_count,):
        raise ValueError(
            "thrusts must have shape "
            f"({QDIMS.motor_count},), got {thrusts.shape}"
        )
    if not np.all(np.isfinite(thrusts)):
        raise ValueError("thrusts must contain only finite values")
    if np.any(thrusts < 0.0):
        raise ValueError("thrusts must be non-negative")


def compute_total_thrust(thrusts: FloatVector) -> float:
    """Return the sum of the four rotor thrusts."""
    _require_thrusts(thrusts)
    return float(np.sum(thrusts))


def compute_torque(
    thrusts: FloatVector,
    arm_length: float,
    yaw_torque_coefficient: float,
    motor_spin_directions: FloatVector,
) -> FloatVector:
    """Return the body torque vector from rotor thrusts."""
    _require_thrusts(thrusts)
    if not np.isfinite(arm_length) or arm_length <= 0.0:
        raise ValueError("arm_length must be finite and positive")
    if not np.isfinite(yaw_torque_coefficient) or yaw_torque_coefficient <= 0.0:
        raise ValueError("yaw_torque_coefficient must be finite and positive")
    if not isinstance(motor_spin_directions, np.ndarray):
        raise TypeError("motor_spin_directions must be a NumPy array")
    if motor_spin_directions.shape != (QDIMS.motor_count,):
        raise ValueError(
            "motor_spin_directions must have shape "
            f"({QDIMS.motor_count},), "
            f"got {motor_spin_directions.shape}"
        )
    if not np.all(np.isin(motor_spin_directions, (-1.0, 1.0))):
        raise ValueError("motor_spin_directions must contain only -1 or 1")

    f1, f2, f3, f4 = thrusts
    return np.array(
        [
            arm_length * (f4 - f2),
            arm_length * (f3 - f1),
            yaw_torque_coefficient * np.dot(motor_spin_directions, thrusts),
        ],
        dtype=np.float64,
    )
