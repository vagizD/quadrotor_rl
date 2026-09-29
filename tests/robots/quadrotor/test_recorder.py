import numpy as np
import pytest

from geometry.common.dimensions import QDIMS
from research.simulator.integrators import (
    EulerIntegrator,
    RK4Integrator,
)
from robots.quadrotor.recorder import record_quadrotor_trajectory
from tests.common import make_quadrotor


@pytest.mark.parametrize("integrator_type", [EulerIntegrator, RK4Integrator])
def test_recorder_aligns_states_and_thrust_transitions(
    integrator_type: type[EulerIntegrator] | type[RK4Integrator],
) -> None:
    quadrotor = make_quadrotor(thrusts=np.full(QDIMS.motor_count, 0.5))
    initial_state = quadrotor.state.to_vector()
    initial_thrusts = quadrotor.thrusts.copy()
    thrust_schedule = np.array([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    target = np.array([0.0, 0.0, 1.0])

    trajectory = record_quadrotor_trajectory(
        quadrotor,
        integrator_type(dt=0.01),
        thrust_schedule,
        target,
    )

    np.testing.assert_allclose(trajectory.times, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(trajectory.states[0], initial_state)
    np.testing.assert_allclose(trajectory.thrusts, thrust_schedule)
    np.testing.assert_allclose(trajectory.target_positions, target)
    np.testing.assert_allclose(quadrotor.state.to_vector(), initial_state)
    np.testing.assert_allclose(quadrotor.thrusts, initial_thrusts)


@pytest.mark.parametrize(
    ("integrator_type", "expected_position_z"),
    [
        # With two Euler steps from rest, p_1 = 0 and p_2 = a * dt^2 because
        # each position update uses the velocity at the start of its step.
        (EulerIntegrator, -9.81 * 0.1**2),
        # RK4 integrates this constant-acceleration free fall exactly over
        # total time 2 * dt: p = 1/2 * a * (2 * dt)^2.
        (RK4Integrator, -0.5 * 9.81 * 0.2**2),
    ],
)
def test_recorder_produces_expected_free_fall_samples(
    integrator_type: type[EulerIntegrator] | type[RK4Integrator],
    expected_position_z: float,
) -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.zeros(QDIMS.motor_count))
    thrust_schedule = np.zeros((2, QDIMS.motor_count))

    trajectory = record_quadrotor_trajectory(
        quadrotor,
        integrator_type(dt=0.1),
        thrust_schedule,
    )

    np.testing.assert_allclose(trajectory.states[-1, 2], expected_position_z)
    np.testing.assert_allclose(trajectory.states[-1, 5], -9.81 * 0.2)
