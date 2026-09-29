import numpy as np
import pytest

from geometry.common.dimensions import QDIMS
from robots.quadrotor import (
    Quadrotor,
    RigidBodyState,
    compute_state_derivative_vector,
)
from research.simulator.integrators import EulerIntegrator, RK4Integrator
from tests.common import make_quadrotor


def advance_quadrotor(
    quadrotor: Quadrotor,
    integrator: EulerIntegrator | RK4Integrator,
    steps: int,
) -> RigidBodyState:
    state_vector = quadrotor.state.to_vector()

    def derivative_function(current: np.ndarray) -> np.ndarray:
        return compute_state_derivative_vector(
            current,
            quadrotor.thrusts,
            quadrotor.params,
        )

    for _ in range(steps):
        state_vector = integrator.step(state_vector, derivative_function)
    return RigidBodyState.from_vector(state_vector)


@pytest.mark.parametrize(
    ("integrator_type", "expected"),
    [
        (EulerIntegrator, np.array([1.2, 2.4])),
        (RK4Integrator, np.array([1.2214, 2.4428])),
    ],
)
def test_integrators_advance_a_known_linear_ode(
    integrator_type: type[EulerIntegrator] | type[RK4Integrator],
    expected: np.ndarray,
) -> None:
    state = np.array([1.0, 2.0])
    integrator = integrator_type(dt=0.1)

    # For x_dot = 2*x, each method uses its own one-step approximation.
    next_state = integrator.step(state, lambda current: 2.0 * current)

    np.testing.assert_allclose(next_state, expected)
    np.testing.assert_allclose(state, [1.0, 2.0])


@pytest.mark.parametrize("integrator_type", [EulerIntegrator, RK4Integrator])
def test_integrators_require_positive_timestep(
    integrator_type: type[EulerIntegrator] | type[RK4Integrator],
) -> None:
    with pytest.raises(ValueError, match="dt"):
        integrator_type(dt=0.0)


def _final_error(
    integrator_type: type[EulerIntegrator] | type[RK4Integrator],
    dt: float,
) -> float:
    state = np.array([1.0])
    integrator = integrator_type(dt=dt)
    for _ in range(round(1.0 / dt)):
        state = integrator.step(state, lambda current: current)
    return abs(state[0] - np.e)


def test_euler_and_rk4_converge_at_their_expected_rates() -> None:
    timesteps = [0.1, 0.05, 0.025]
    euler_errors = [_final_error(EulerIntegrator, dt) for dt in timesteps]
    rk4_errors = [_final_error(RK4Integrator, dt) for dt in timesteps]

    assert euler_errors[2] < euler_errors[1] < euler_errors[0]
    assert rk4_errors[2] < rk4_errors[1] < rk4_errors[0]
    assert rk4_errors[0] < euler_errors[2]
    assert rk4_errors[0] / rk4_errors[1] > 8.0


@pytest.mark.parametrize("integrator_type", [EulerIntegrator, RK4Integrator])
def test_quadrotor_hover_stays_at_equilibrium(
    integrator_type: type[EulerIntegrator] | type[RK4Integrator],
) -> None:
    quadrotor = make_quadrotor()
    hover_thrust = (
        quadrotor.params.mass
        * quadrotor.params.gravity
        / QDIMS.motor_count
    )
    quadrotor.set_thrusts(np.full(QDIMS.motor_count, hover_thrust))
    initial_vector = quadrotor.state.to_vector()

    next_state = advance_quadrotor(
        quadrotor,
        integrator_type(dt=0.01),
        steps=10,
    )

    np.testing.assert_allclose(next_state.to_vector(), initial_vector, atol=1e-12)
    np.testing.assert_allclose(quadrotor.state.to_vector(), initial_vector)


@pytest.mark.parametrize(
    ("integrator_type", "expected_position_z"),
    [
        (EulerIntegrator, -9.81 * 0.01**2 * 10 * 9 / 2),
        (RK4Integrator, -0.5 * 9.81 * (0.01 * 10) ** 2),
    ],
)
def test_quadrotor_free_fall_matches_integrator(
    integrator_type: type[EulerIntegrator] | type[RK4Integrator],
    expected_position_z: float,
) -> None:
    # Euler
    # p_z = -g * dt² * N(N - 1) / 2
    # x_1:  p = 0, v = a * dt  |  x_2: p = a * dt², v = 2a * dt  |  x_3: p = (1 + 2)a * dt², v = 3a * dt

    # RK4
    # p_z = -g * dt² * N² / 2
    # p_t = 1/2 * a * total_time² = 1/2 * a * (dt * N)²
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.zeros(QDIMS.motor_count))
    dt = 0.01
    steps = 10

    next_state = advance_quadrotor(quadrotor, integrator_type(dt=dt), steps)
    expected_velocity_z = -quadrotor.params.gravity * dt * steps

    np.testing.assert_allclose(next_state.position, [0.0, 0.0, expected_position_z])
    np.testing.assert_allclose(next_state.velocity, [0.0, 0.0, expected_velocity_z])
    np.testing.assert_allclose(next_state.quaternion.as_array(), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(next_state.angular_velocity, np.zeros(3))
