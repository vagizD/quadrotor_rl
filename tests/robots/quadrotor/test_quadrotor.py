import numpy as np
import pytest

from common.dimensions import QDIMS
from tests.common import make_quadrotor


def test_quadrotor_delegates_motor_calculations() -> None:
    quadrotor = make_quadrotor()
    quadrotor.set_thrusts(np.arange(1.0, QDIMS.motor_count + 1.0))

    assert quadrotor.compute_total_thrust() == 10.0
    np.testing.assert_allclose(quadrotor.compute_torque(), [0.4, 0.4, -0.2])


def test_set_thrusts_copies_and_enforces_maximum() -> None:
    quadrotor = make_quadrotor()
    thrusts = np.array([1.0, 1.0, 1.0, 1.0])
    quadrotor.set_thrusts(thrusts)
    thrusts[0] = 4.0

    np.testing.assert_allclose(quadrotor.thrusts, [1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="max_thrust"):
        quadrotor.set_thrusts(np.full(QDIMS.motor_count, 5.1))
