import numpy as np
import pytest

from common.dimensions import QDIMS
from simulation.trajectory import Trajectory


def make_trajectory(target_positions: np.ndarray | None = None) -> Trajectory:
    return Trajectory(
        times=np.array([0.0, 0.1, 0.2]),
        states=np.zeros((3, QDIMS.state_dim)),
        thrusts=np.ones((2, QDIMS.motor_count)),
        target_positions=target_positions,
    )


def test_trajectory_aligns_samples_and_transitions() -> None:
    trajectory = make_trajectory()

    assert trajectory.times.shape == (3,)
    assert trajectory.states.shape == (3, QDIMS.state_dim)
    assert trajectory.thrusts.shape == (2, QDIMS.motor_count)
    assert trajectory.step_count == 2


def test_trajectory_accepts_fixed_or_time_varying_targets() -> None:
    np.testing.assert_allclose(
        make_trajectory(np.array([1.0, 2.0, 3.0])).target_positions,
        [1.0, 2.0, 3.0],
    )
    time_varying = np.zeros((3, 3))

    assert make_trajectory(time_varying).target_positions.shape == (3, 3)


def test_trajectory_copies_recorded_arrays() -> None:
    times = np.array([0.0, 0.1])
    states = np.zeros((2, QDIMS.state_dim))
    thrusts = np.ones((1, QDIMS.motor_count))
    trajectory = Trajectory(times, states, thrusts)
    times[0] = 99.0
    states[0, 0] = 99.0
    thrusts[0, 0] = 99.0

    np.testing.assert_allclose(trajectory.times, [0.0, 0.1])
    assert trajectory.states[0, 0] == 0.0
    assert trajectory.thrusts[0, 0] == 1.0


def test_trajectory_rejects_misaligned_or_invalid_data() -> None:
    with pytest.raises(ValueError, match="states"):
        Trajectory(
            np.array([0.0, 0.1]),
            np.zeros((1, QDIMS.state_dim)),
            np.ones((1, QDIMS.motor_count)),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory(
            np.array([0.0, 0.0]),
            np.zeros((2, QDIMS.state_dim)),
            np.ones((1, QDIMS.motor_count)),
        )
    with pytest.raises(ValueError, match="target_positions"):
        make_trajectory(np.zeros((2, 3)))
