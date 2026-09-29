"""Quadrotor rigid-body derivative calculations."""

import numpy as np

from geometry.common.types import FloatVector
from geometry.math.rotation import Quaternion

from .motors import (
    compute_torque,
    compute_total_thrust,
)
from .params import QuadrotorParams
from geometry.state import RigidBodyState, RigidBodyStateDerivative


def _require_vector(value: FloatVector, name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def compute_position_derivative(state: RigidBodyState) -> FloatVector:
    """Return the position derivative for the current state."""
    return state.velocity.copy()


def compute_quaternion_derivative(
    quaternion: Quaternion,
    angular_velocity: FloatVector,
) -> FloatVector:
    """Return the quaternion derivative."""
    if not isinstance(quaternion, Quaternion):
        raise TypeError("quaternion must be a Quaternion")
    if not np.isclose(quaternion.norm(), 1.0):
        raise ValueError("quaternion must have unit norm")
    _require_vector(angular_velocity, "angular_velocity")

    pure_angular_velocity = Quaternion(
        0.0,
        angular_velocity[0],
        angular_velocity[1],
        angular_velocity[2],
        normalize=False,
    )
    return 0.5 * (quaternion * pure_angular_velocity).as_array()


def compute_angular_acceleration(
    angular_velocity: FloatVector,
    torque: FloatVector,
    params: QuadrotorParams,
    external_torque: FloatVector | None = None,
) -> FloatVector:
    """Return angular acceleration including rigid-body coupling."""
    _require_vector(angular_velocity, "angular_velocity")
    _require_vector(torque, "torque")
    angular_momentum = params.inertia @ angular_velocity
    gyroscopic_term = np.cross(angular_velocity, angular_momentum)
    net_torque = torque - gyroscopic_term
    if external_torque is not None:
        _require_vector(external_torque, "external_torque")
        net_torque += external_torque
    return np.linalg.solve(params.inertia, net_torque)


def compute_gravity_acceleration(params: QuadrotorParams) -> FloatVector:
    """Return the world-frame gravity acceleration."""
    return np.array([0.0, 0.0, -params.gravity], dtype=np.float64)


def compute_external_force_acceleration(
    external_force: FloatVector,
    params: QuadrotorParams,
) -> FloatVector:
    """Return world-frame acceleration caused by an external force."""
    _require_vector(external_force, "external_force")
    return external_force / params.mass



def compute_linear_acceleration(
    state: RigidBodyState,
    thrusts: FloatVector,
    params: QuadrotorParams,
    external_force: FloatVector | None = None,
) -> FloatVector:
    """Return world-frame acceleration from gravity and rotor thrust."""
    total_thrust = compute_total_thrust(thrusts)
    thrust_body = np.array([0.0, 0.0, total_thrust], dtype=np.float64)
    thrust_world = state.quaternion.to_rotation_matrix() @ thrust_body
    acceleration = compute_gravity_acceleration(params) + thrust_world / params.mass
    if external_force is not None:
        acceleration += compute_external_force_acceleration(external_force, params)
    return acceleration


def compute_state_derivative(
    state: RigidBodyState,
    thrusts: FloatVector,
    params: QuadrotorParams,
    external_force: FloatVector | None = None,
    external_torque: FloatVector | None = None,
) -> RigidBodyStateDerivative:
    """Return all state derivatives from one current state and action."""
    torque = compute_torque(
        thrusts,
        params.arm_length,
        params.yaw_torque_coefficient,
        params.motor_spin_directions,
    )
    return RigidBodyStateDerivative(
        position=compute_position_derivative(state),
        velocity=compute_linear_acceleration(state, thrusts, params, external_force),
        quaternion=compute_quaternion_derivative(
            state.quaternion,
            state.angular_velocity,
        ),
        angular_velocity=compute_angular_acceleration(
            state.angular_velocity,
            torque,
            params,
            external_torque,
        ),
    )


def compute_state_derivative_vector(
    state_vector: FloatVector,
    thrusts: FloatVector,
    params: QuadrotorParams,
    external_force: FloatVector | None = None,
    external_torque: FloatVector | None = None,
) -> FloatVector:
    """Return the full derivative as a flat vector for integrator callbacks."""
    return compute_state_derivative(
        RigidBodyState.from_vector(state_vector),
        thrusts,
        params,
        external_force,
        external_torque,
    ).to_vector()
