import numpy as np
import pytest

from geometry.common.dimensions import QDIMS
from geometry.math.rotation import Quaternion
from robots.quadrotor.dynamics import (
    compute_angular_acceleration,
    compute_external_force_acceleration,
    compute_gravity_acceleration,
    compute_linear_acceleration,
    compute_position_derivative,
    compute_quaternion_derivative,
    compute_state_derivative,
    compute_torque,
)
from tests.common import make_quadrotor, make_state


def test_position_derivative_equals_velocity_without_aliasing() -> None:
    state = make_state(velocity=np.array([1.0, 2.0, 3.0]))
    derivative = compute_position_derivative(state)
    derivative[0] = 99.0

    np.testing.assert_allclose(derivative, [99.0, 2.0, 3.0])
    np.testing.assert_allclose(state.velocity, [1.0, 2.0, 3.0])


def test_quadrotor_position_derivative_delegates_to_pure_function() -> None:
    state = make_state(velocity=np.array([1.0, 2.0, 3.0]))
    quadrotor = make_quadrotor(state)

    np.testing.assert_allclose(quadrotor.compute_position_derivative(), state.velocity)


def test_quaternion_derivative_uses_body_angular_velocity() -> None:
    angular_velocity = np.array([1.0, 2.0, 3.0])
    state = make_state(angular_velocity=angular_velocity)
    quadrotor = make_quadrotor(state)

    # [0, omega] is a pure quaternion used in the derivative equation; q_dot
    # is therefore a raw four-vector, not another orientation quaternion.
    derivative = compute_quaternion_derivative(Quaternion.identity(), angular_velocity)

    np.testing.assert_allclose(derivative, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(quadrotor.compute_quaternion_derivative(), derivative)


def test_state_derivative_assembles_all_rates_from_one_state_and_action() -> None:
    state = make_state(
        velocity=np.array([1.0, 2.0, 3.0]),
        angular_velocity=np.array([1.0, 2.0, 3.0]),
    )
    quadrotor = make_quadrotor(state)
    thrusts = np.arange(1.0, QDIMS.motor_count + 1.0)
    external_force = np.array([0.5, -1.0, 2.0])
    quadrotor.set_thrusts(thrusts)

    # Asymmetric thrusts and nonzero angular velocity exercise both torque terms
    # while every rate is still evaluated from this same state and action.
    derivative = compute_state_derivative(
        state,
        thrusts,
        quadrotor.params,
        external_force,
    )
    torque = compute_torque(
        thrusts,
        quadrotor.params.arm_length,
        quadrotor.params.yaw_torque_coefficient,
        quadrotor.params.motor_spin_directions,
    )

    np.testing.assert_allclose(derivative.position, state.velocity)
    np.testing.assert_allclose(
        derivative.velocity,
        compute_linear_acceleration(
            state,
            thrusts,
            quadrotor.params,
            external_force,
        ),
    )
    np.testing.assert_allclose(
        derivative.quaternion,
        compute_quaternion_derivative(state.quaternion, state.angular_velocity),
    )
    np.testing.assert_allclose(
        derivative.angular_velocity,
        compute_angular_acceleration(
            state.angular_velocity,
            torque,
            quadrotor.params,
        ),
    )
    facade_derivative = quadrotor.compute_state_derivative(external_force)
    np.testing.assert_allclose(facade_derivative.position, derivative.position)
    np.testing.assert_allclose(facade_derivative.velocity, derivative.velocity)
    np.testing.assert_allclose(facade_derivative.quaternion, derivative.quaternion)
    np.testing.assert_allclose(
        facade_derivative.angular_velocity,
        derivative.angular_velocity,
    )


def test_hover_is_a_full_state_derivative_equilibrium() -> None:
    state = make_state()
    quadrotor = make_quadrotor(state)
    hover_thrust = (
        quadrotor.params.mass
        * quadrotor.params.gravity
        / QDIMS.motor_count
    )
    quadrotor.set_thrusts(np.full(QDIMS.motor_count, hover_thrust))

    # Equal thrust cancels gravity and the symmetric rotor layout cancels torque.
    derivative = compute_state_derivative(
        state,
        quadrotor.thrusts,
        quadrotor.params,
    )

    np.testing.assert_allclose(derivative.position, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(derivative.velocity, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(
        derivative.quaternion,
        np.zeros(QDIMS.quaternion_dim),
        atol=1e-12,
    )
    np.testing.assert_allclose(derivative.angular_velocity, np.zeros(3), atol=1e-12)
    facade_derivative = quadrotor.compute_state_derivative()
    np.testing.assert_allclose(facade_derivative.position, derivative.position)
    np.testing.assert_allclose(facade_derivative.velocity, derivative.velocity)
    np.testing.assert_allclose(facade_derivative.quaternion, derivative.quaternion)
    np.testing.assert_allclose(
        facade_derivative.angular_velocity,
        derivative.angular_velocity,
    )


def test_gravity_acceleration_points_downward() -> None:
    quadrotor = make_quadrotor()

    np.testing.assert_allclose(
        compute_gravity_acceleration(quadrotor.params),
        [0.0, 0.0, -9.81],
    )


def test_quadrotor_gravity_acceleration_delegates_to_pure_function() -> None:
    quadrotor = make_quadrotor()

    np.testing.assert_allclose(
        quadrotor.compute_gravity_acceleration(),
        [0.0, 0.0, -9.81],
    )


def test_external_force_changes_acceleration_by_force_over_mass() -> None:
    quadrotor = make_quadrotor()
    external_force = np.array([1.0, 2.0, 3.0])
    baseline = quadrotor.compute_linear_acceleration()

    acceleration = compute_linear_acceleration(
        quadrotor.state,
        quadrotor.thrusts,
        quadrotor.params,
        external_force,
    )

    np.testing.assert_allclose(acceleration - baseline, external_force)
    np.testing.assert_allclose(
        quadrotor.compute_external_force_acceleration(external_force),
        external_force,
    )


def test_external_force_requires_finite_three_vector() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_external_force_acceleration(np.ones(2), make_quadrotor().params)
    with pytest.raises(ValueError, match="finite"):
        compute_external_force_acceleration(
            np.array([1.0, np.nan, 0.0]),
            make_quadrotor().params,
        )


def test_zero_rate_torque_response_is_inertia_inverse_times_torque() -> None:
    quadrotor = make_quadrotor()
    angular_velocity = np.zeros(3)
    torque = np.array([0.01, -0.02, 0.04])

    acceleration = compute_angular_acceleration(
        angular_velocity,
        torque,
        quadrotor.params,
    )

    np.testing.assert_allclose(acceleration, [1.0, -2.0, 2.0])
    np.testing.assert_allclose(
        quadrotor.compute_angular_acceleration(angular_velocity, torque),
        acceleration,
    )


def test_rigid_body_angular_acceleration_includes_cross_term() -> None:
    quadrotor = make_quadrotor()
    angular_velocity = np.array([1.0, 2.0, 3.0])
    torque = np.zeros(3)
    # For this diagonal inertia, omega x (J*omega) = [0.06, -0.03, 0].

    acceleration = compute_angular_acceleration(
        angular_velocity,
        torque,
        quadrotor.params,
    )

    np.testing.assert_allclose(acceleration, [-6.0, 3.0, 0.0])
    np.testing.assert_allclose(
        quadrotor.compute_angular_acceleration(angular_velocity, torque),
        acceleration,
    )
