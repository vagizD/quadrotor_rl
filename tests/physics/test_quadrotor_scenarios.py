import numpy as np

from common.dimensions import QDIMS
from geometry.rotation import Quaternion
from tests.common import make_quadrotor, make_state


def test_free_fall_has_only_downward_gravity_acceleration() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.zeros(QDIMS.motor_count))

    derivative = quadrotor.compute_state_derivative()

    np.testing.assert_allclose(derivative.position, np.zeros(3))
    np.testing.assert_allclose(derivative.velocity, [0.0, 0.0, -9.81])
    np.testing.assert_allclose(
        derivative.quaternion,
        np.zeros(QDIMS.quaternion_dim),
    )
    np.testing.assert_allclose(derivative.angular_velocity, np.zeros(3))


def test_hover_equilibrium_has_zero_full_state_derivative() -> None:
    quadrotor = make_quadrotor()
    hover_thrust = (
        quadrotor.params.mass
        * quadrotor.params.gravity
        / QDIMS.motor_count
    )
    quadrotor.set_thrusts(np.full(QDIMS.motor_count, hover_thrust))

    derivative = quadrotor.compute_state_derivative()

    np.testing.assert_allclose(derivative.position, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(derivative.velocity, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(
        derivative.quaternion,
        np.zeros(QDIMS.quaternion_dim),
        atol=1e-12,
    )
    np.testing.assert_allclose(derivative.angular_velocity, np.zeros(3), atol=1e-12)


def test_vertical_climb_has_positive_world_z_acceleration() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(
        np.full(
            QDIMS.motor_count,
            (quadrotor.params.mass * 9.81 + 1.0) / QDIMS.motor_count,
        )
    )

    np.testing.assert_allclose(quadrotor.compute_linear_acceleration(), [0.0, 0.0, 1.0])


def test_vertical_descent_has_negative_world_z_acceleration() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(
        np.full(
            QDIMS.motor_count,
            (quadrotor.params.mass * 9.81 - 1.0) / QDIMS.motor_count,
        )
    )

    np.testing.assert_allclose(quadrotor.compute_linear_acceleration(), [0.0, 0.0, -1.0])


def test_pure_roll_configuration_has_only_roll_torque() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.array([2.0, 1.0, 2.0, 3.0]))

    # f4 - f2 creates roll; the other thrust differences and yaw signs cancel.
    np.testing.assert_allclose(quadrotor.compute_torque(), [0.4, 0.0, 0.0])


def test_pure_pitch_configuration_has_only_pitch_torque() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.array([1.0, 2.0, 3.0, 2.0]))

    # f3 - f1 creates pitch; the other thrust differences and yaw signs cancel.
    np.testing.assert_allclose(quadrotor.compute_torque(), [0.0, 0.4, 0.0])


def test_pure_yaw_configuration_has_only_yaw_torque() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.array([2.0, 1.0, 2.0, 1.0]))

    # Opposite spin groups have different thrust while arm torques cancel.
    np.testing.assert_allclose(quadrotor.compute_torque(), [0.0, 0.0, 0.2])


def test_tilted_thrust_produces_expected_horizontal_acceleration() -> None:
    angle = np.pi / 2.0
    quadrotor = make_quadrotor(
        state=make_state(
            quaternion=Quaternion(
                np.cos(angle / 2.0),
                0.0,
                np.sin(angle / 2.0),
                0.0,
            )
        )
    )
    hover_thrust = (
        quadrotor.params.mass
        * quadrotor.params.gravity
        / QDIMS.motor_count
    )
    quadrotor.set_thrusts(np.full(QDIMS.motor_count, hover_thrust))

    # A positive 90-degree pitch maps +z_B thrust to +x_W, while gravity stays -z_W.
    np.testing.assert_allclose(
        quadrotor.compute_linear_acceleration(),
        [9.81, 0.0, -9.81],
        atol=1e-12,
    )
