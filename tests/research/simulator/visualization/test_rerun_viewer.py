import numpy as np
import pytest

from geometry.common.dimensions import QDIMS
from research.pipeline.control.rl.ppo import PPOTrajectory
from research.simulator.integrators import EulerIntegrator, Trajectory
from robots.quadrotor.recorder import record_quadrotor_trajectory
from tests.common import make_quadrotor
from research.simulator.visualization.rerun_viewer import log_trajectory

pytest.importorskip("rerun")


def make_recorded_trajectory() -> Trajectory:
    quadrotor = make_quadrotor()
    hover_thrust = quadrotor.params.gravity / QDIMS.motor_count
    thrust_schedule = np.full(
        (3, QDIMS.motor_count),
        hover_thrust,
    )
    targets = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.1], [0.0, 0.0, 1.2], [0.0, 0.0, 1.3]]
    )
    return record_quadrotor_trajectory(
        quadrotor,
        EulerIntegrator(dt=0.01),
        thrust_schedule,
        targets,
    )


def test_rerun_logger_saves_time_varying_target_recording(tmp_path) -> None:
    trajectory = make_recorded_trajectory()
    recording_path = tmp_path / "diagnostics.rrd"

    saved_path = log_trajectory(trajectory, 0.2, recording_path)

    assert saved_path == recording_path
    assert recording_path.exists() and recording_path.stat().st_size > 0

    import rerun.experimental as rrx

    assert rrx.RrdReader(recording_path).blueprints()


def test_rerun_logger_requires_positive_arm_length(tmp_path) -> None:
    trajectory = make_recorded_trajectory()

    with pytest.raises(ValueError, match="arm_length"):
        log_trajectory(trajectory, 0.0, tmp_path / "invalid.rrd")


def make_ppo_trajectory() -> PPOTrajectory:
    trajectory = make_recorded_trajectory()
    return PPOTrajectory(
        times=trajectory.times,
        states=trajectory.states,
        thrusts=trajectory.thrusts,
        target_positions=trajectory.target_positions,
        observations=np.zeros((3, QDIMS.observation_dim)),
        rewards=np.array([1.0, 0.5, -2.0]),
        values=np.array([0.2, 0.3, 0.4]),
        log_probabilities=np.array([-1.0, -1.1, -1.2]),
        terminated=np.array([False, False, True]),
        truncated=np.array([False, False, False]),
        returns=np.array([0.5, -0.5, -2.0]),
        advantages=np.array([0.3, -0.8, -2.4]),
    )


def test_rerun_logger_accepts_ppo_trajectory_and_samples_physics(tmp_path) -> None:
    trajectory = make_ppo_trajectory()
    recording_path = tmp_path / "ppo-physics.rrd"

    saved_path = log_trajectory(
        trajectory,
        0.2,
        recording_path,
        step_stride=2,
    )

    assert saved_path == recording_path
    assert recording_path.exists() and recording_path.stat().st_size > 0

    import rerun.experimental as rrx

    assert rrx.RrdReader(recording_path).blueprints()


def test_rerun_logger_requires_positive_step_stride() -> None:
    with pytest.raises(ValueError, match="step_stride"):
        log_trajectory(make_ppo_trajectory(), 0.2, step_stride=0)
