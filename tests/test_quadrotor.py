import numpy as np
import pytest

from geometry.rotation import Quaternion
from robots.quadrotor import Quadrotor, QuadrotorParams, QuadrotorState


def make_quadrotor() -> Quadrotor:
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
        quaternion=Quaternion.identity(),
        angular_velocity=np.zeros(3),
    )
    return Quadrotor(params, state)


def test_quadrotor_delegates_motor_calculations() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.array([1.0, 2.0, 3.0, 4.0]))

    assert quadrotor.compute_total_thrust() == 10.0
    np.testing.assert_allclose(quadrotor.compute_torque(), [0.4, 0.4, -0.2])


def test_set_thrusts_copies_and_enforces_maximum() -> None:
    quadrotor = make_quadrotor()
    thrusts = np.array([1.0, 1.0, 1.0, 1.0])
    quadrotor.set_thrusts(thrusts)
    thrusts[0] = 4.0

    np.testing.assert_allclose(quadrotor.thrusts, [1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="max_thrust"):
        quadrotor.set_thrusts(np.full(4, 5.1))
