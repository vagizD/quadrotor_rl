import numpy as np
import pytest

from robots.quadrotor.motors import (
    compute_torque,
    compute_total_thrust,
)

MOTOR_SPIN_DIRECTIONS = np.array([1.0, -1.0, 1.0, -1.0])


def test_total_thrust_is_sum_of_four_rotors() -> None:
    thrusts = np.array([1.0, 2.0, 3.0, 4.0])

    assert compute_total_thrust(thrusts) == 10.0


@pytest.mark.parametrize("thrusts", [np.ones(3), np.ones((2, 2))])
def test_total_thrust_requires_four_element_vector(thrusts: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_total_thrust(thrusts)


def test_total_thrust_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_total_thrust(np.array([1.0, 1.0, -1.0, 1.0]))


def test_torque_matches_motor_geometry_and_spin_signs() -> None:
    torque = compute_torque(
        np.array([1.0, 2.0, 3.0, 4.0]),
        0.2,
        0.1,
        MOTOR_SPIN_DIRECTIONS,
    )

    np.testing.assert_allclose(torque, [0.4, 0.4, -0.2])


def test_equal_thrusts_produce_no_torque() -> None:
    torque = compute_torque(np.ones(4), 0.2, 0.1, MOTOR_SPIN_DIRECTIONS)

    np.testing.assert_allclose(torque, [0.0, 0.0, 0.0])


def test_torque_rejects_invalid_arm_length() -> None:
    with pytest.raises(ValueError, match="arm_length"):
        compute_torque(np.ones(4), 0.0, 0.1, MOTOR_SPIN_DIRECTIONS)


@pytest.mark.parametrize(
    ("thrusts", "expected"),
    [
        (np.array([2.0, 0.0, 2.0, 0.0]), 0.4),
        (np.array([0.0, 2.0, 0.0, 2.0]), -0.4),
    ],
)
def test_torque_uses_rotor_spin_sign(
    thrusts: np.ndarray,
    expected: float,
) -> None:
    assert compute_torque(thrusts, 0.2, 0.1, MOTOR_SPIN_DIRECTIONS)[2] == expected


def test_torque_rejects_invalid_coefficient() -> None:
    with pytest.raises(ValueError, match="yaw_torque_coefficient"):
        compute_torque(np.ones(4), 0.2, 0.0, MOTOR_SPIN_DIRECTIONS)


def test_torque_rejects_invalid_motor_spin_directions() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_torque(np.ones(4), 0.2, 0.1, np.ones(3))
    with pytest.raises(ValueError, match="only -1 or 1"):
        compute_torque(
            np.ones(4),
            0.2,
            0.1,
            np.array([1.0, 0.0, 1.0, -1.0]),
        )
