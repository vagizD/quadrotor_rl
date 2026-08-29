import numpy as np

from geometry.rotation import Quaternion
from robots.quadrotor import Quadrotor, QuadrotorParams, QuadrotorState


def make_quadrotor(quaternion: Quaternion | None = None) -> Quadrotor:
    params = QuadrotorParams(
        mass=1.0,
        arm_length=0.2,
        inertia=np.diag([0.01, 0.01, 0.02]),
        yaw_torque_coefficient=0.1,
        max_thrust=5.0,
    )
    state = QuadrotorState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        quaternion=Quaternion.identity() if quaternion is None else quaternion,
        angular_velocity=np.zeros(3),
    )
    return Quadrotor(params, state)


def test_free_fall_has_only_downward_gravity_acceleration() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.zeros(4))

    derivative = quadrotor.compute_state_derivative()

    np.testing.assert_allclose(derivative.position, np.zeros(3))
    np.testing.assert_allclose(derivative.velocity, [0.0, 0.0, -9.81])
    np.testing.assert_allclose(derivative.quaternion, np.zeros(4))
    np.testing.assert_allclose(derivative.angular_velocity, np.zeros(3))


def test_hover_equilibrium_has_zero_full_state_derivative() -> None:
    quadrotor = make_quadrotor()
    hover_thrust = quadrotor.params.mass * quadrotor.params.gravity / 4.0
    quadrotor.set_thrusts(np.full(4, hover_thrust))

    derivative = quadrotor.compute_state_derivative()

    np.testing.assert_allclose(derivative.position, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(derivative.velocity, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(derivative.quaternion, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(derivative.angular_velocity, np.zeros(3), atol=1e-12)


def test_vertical_climb_has_positive_world_z_acceleration() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.full(4, (quadrotor.params.mass * 9.81 + 1.0) / 4.0))

    np.testing.assert_allclose(quadrotor.compute_linear_acceleration(), [0.0, 0.0, 1.0])


def test_vertical_descent_has_negative_world_z_acceleration() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.full(4, (quadrotor.params.mass * 9.81 - 1.0) / 4.0))

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
        Quaternion(
            np.cos(angle / 2.0),
            0.0,
            np.sin(angle / 2.0),
            0.0,
        )
    )
    hover_thrust = quadrotor.params.mass * quadrotor.params.gravity / 4.0
    quadrotor.set_thrusts(np.full(4, hover_thrust))

    # A positive 90-degree pitch maps +z_B thrust to +x_W, while gravity stays -z_W.
    np.testing.assert_allclose(
        quadrotor.compute_linear_acceleration(),
        [9.81, 0.0, -9.81],
        atol=1e-12,
    )
