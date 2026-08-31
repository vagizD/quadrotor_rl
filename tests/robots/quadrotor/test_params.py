import numpy as np
import pytest

from robots.quadrotor.params import QuadrotorParams


def make_params(**overrides: object) -> QuadrotorParams:
    values: dict[str, object] = {
        "mass": 1.0,
        "arm_length": 0.2,
        "inertia": np.diag([0.01, 0.01, 0.02]),
        "yaw_torque_coefficient": 0.01,
        "max_thrust": 10.0,
    }
    values.update(overrides)
    return QuadrotorParams(**values)


def test_valid_parameters_are_stored() -> None:
    params = make_params()

    assert params.mass == 1.0
    assert params.gravity == 9.81
    np.testing.assert_allclose(params.inertia, np.diag([0.01, 0.01, 0.02]))
    np.testing.assert_allclose(
        params.motor_spin_directions,
        [1.0, -1.0, 1.0, -1.0],
    )


@pytest.mark.parametrize(
    "field",
    ["mass", "arm_length", "yaw_torque_coefficient", "max_thrust", "gravity"],
)
def test_scalar_parameters_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_params(**{field: 0.0})


def test_inertia_must_be_positive_diagonal_three_by_three_matrix() -> None:
    with pytest.raises(ValueError, match="diagonal"):
        make_params(inertia=np.ones((3, 3)))

    with pytest.raises(ValueError, match="shape"):
        make_params(inertia=np.ones(3))


def test_motor_spin_directions_must_be_four_signs() -> None:
    with pytest.raises(ValueError, match="shape"):
        make_params(motor_spin_directions=np.ones(3))
    with pytest.raises(ValueError, match="only -1 or 1"):
        make_params(motor_spin_directions=np.array([1.0, 0.0, 1.0, -1.0]))
