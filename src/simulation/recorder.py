"""Physics trajectory recording helpers."""

import numpy as np

from common.dimensions import QDIMS
from common.types import FloatMatrix, FloatVector
from robots.quadrotor import (
    Quadrotor,
    QuadrotorState,
    compute_state_derivative_vector,
)

from .integrators import EulerIntegrator, RK4Integrator
from .trajectory import Trajectory


def _require_thrust_schedule(thrust_schedule: FloatMatrix) -> None:
    if not isinstance(thrust_schedule, np.ndarray):
        raise TypeError("thrust_schedule must be a NumPy array")
    if (
        thrust_schedule.ndim != 2
        or thrust_schedule.shape[1] != QDIMS.motor_count
    ):
        raise ValueError(
            "thrust_schedule must have shape "
            f"(N, {QDIMS.motor_count}), "
            f"got {thrust_schedule.shape}"
        )
    if not np.all(np.isfinite(thrust_schedule)):
        raise ValueError("thrust_schedule must contain only finite values")
    if np.any(thrust_schedule < 0.0):
        raise ValueError("thrust_schedule must be non-negative")


def record_quadrotor_trajectory(
    quadrotor: Quadrotor,
    integrator: EulerIntegrator | RK4Integrator,
    thrust_schedule: FloatMatrix,
    target_positions: FloatVector | FloatMatrix | None = None,
) -> Trajectory:
    """Record a quadrotor trajectory for a fixed thrust schedule."""
    if not isinstance(quadrotor, Quadrotor):
        raise TypeError("quadrotor must be a Quadrotor")
    if not isinstance(integrator, (EulerIntegrator, RK4Integrator)):
        raise TypeError("integrator must be EulerIntegrator or RK4Integrator")
    _require_thrust_schedule(thrust_schedule)

    state_vector = quadrotor.state.to_vector()
    states = [state_vector.copy()]

    for thrusts in thrust_schedule:
        def derivative_function(current: FloatVector) -> FloatVector:
            return compute_state_derivative_vector(
                current,
                thrusts,
                quadrotor.params,
            )

        state_vector = integrator.step(state_vector, derivative_function)
        states.append(QuadrotorState.from_vector(state_vector).to_vector())

    times = np.arange(len(states), dtype=np.float64) * integrator.dt
    return Trajectory(
        times=times,
        states=np.array(states, dtype=np.float64),
        thrusts=thrust_schedule,
        target_positions=target_positions,
    )
