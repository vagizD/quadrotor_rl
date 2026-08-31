"""Stateful quadrotor model façade."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from common.dimensions import QDIMS
from common.types import FloatVector
from .dynamics import compute_external_force_acceleration as _compute_external_force_acceleration
from .dynamics import compute_gravity_acceleration as _compute_gravity_acceleration
from .dynamics import compute_linear_acceleration as _compute_linear_acceleration
from .dynamics import compute_angular_acceleration as _compute_angular_acceleration
from .dynamics import compute_position_derivative as _compute_position_derivative
from .dynamics import compute_quaternion_derivative as _compute_quaternion_derivative
from .dynamics import compute_state_derivative as _compute_state_derivative
from .motors import (
    compute_torque as _compute_torque,
    compute_total_thrust as _compute_total_thrust,
)
from .params import QuadrotorParams
from .state import QuadrotorState, QuadrotorStateDerivative


@dataclass
class Quadrotor:
    """Own quadrotor state, parameters, and current thrust command."""

    params: QuadrotorParams
    state: QuadrotorState
    thrusts: FloatVector = field(
        default_factory=lambda: np.zeros(QDIMS.motor_count),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.params, QuadrotorParams):
            raise TypeError("params must be QuadrotorParams")
        if not isinstance(self.state, QuadrotorState):
            raise TypeError("state must be QuadrotorState")
        self.set_thrusts(self.thrusts)

    def set_thrusts(self, thrusts: ArrayLike) -> None:
        """Validate and store the current per-rotor thrust command."""
        value = np.array(thrusts, dtype=np.float64, copy=True)
        _compute_total_thrust(value)
        if np.any(value > self.params.max_thrust):
            raise ValueError("thrusts cannot exceed params.max_thrust")
        self.thrusts = value

    def compute_total_thrust(self) -> float:
        return _compute_total_thrust(self.thrusts)

    def compute_torque(self) -> FloatVector:
        return _compute_torque(
            self.thrusts,
            self.params.arm_length,
            self.params.yaw_torque_coefficient,
            self.params.motor_spin_directions,
        )

    def compute_position_derivative(self) -> FloatVector:
        return _compute_position_derivative(self.state)

    def compute_quaternion_derivative(self) -> FloatVector:
        return _compute_quaternion_derivative(
            self.state.quaternion,
            self.state.angular_velocity,
        )

    def compute_state_derivative(
        self,
        external_force: FloatVector | None = None,
    ) -> QuadrotorStateDerivative:
        return _compute_state_derivative(
            self.state,
            self.thrusts,
            self.params,
            external_force,
        )

    def compute_gravity_acceleration(self) -> FloatVector:
        return _compute_gravity_acceleration(self.params)

    def compute_external_force_acceleration(self, external_force: FloatVector) -> FloatVector:
        return _compute_external_force_acceleration(external_force, self.params)

    def compute_angular_acceleration(
        self,
        angular_velocity: FloatVector,
        torque: FloatVector,
    ) -> FloatVector:
        return _compute_angular_acceleration(
            angular_velocity,
            torque,
            self.params,
        )

    def compute_linear_acceleration(
        self,
        external_force: FloatVector | None = None,
    ) -> FloatVector:
        return _compute_linear_acceleration(
            self.state,
            self.thrusts,
            self.params,
            external_force,
        )
