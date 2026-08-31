"""Optional Rerun playback for recorded quadrotor trajectories."""

from numbers import Integral
from pathlib import Path

import numpy as np

from common.dimensions import QDIMS
from simulation.trajectory import Trajectory


def _require_rerun():
    try:
        import rerun as rr
    except ModuleNotFoundError as error:
        raise ImportError(
            "Rerun support requires the visualization extra: "
            "uv sync --extra visualization"
        ) from error
    return rr


def _default_blueprint():
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="Quadrotor", origin="/world"),
            rrb.TimeSeriesView(
                name="Motor thrusts",
                origin="/world/quadrotor/motors",
            ),
        ),
        collapse_panels=True,
    )


def _require_arm_length(arm_length: float) -> None:
    if not np.isfinite(arm_length) or arm_length <= 0.0:
        raise ValueError("arm_length must be finite and positive")


def _quaternion_xyzw(state: np.ndarray) -> np.ndarray:
    # The simulator stores scalar-first [w, x, y, z]; Rerun accepts [x, y, z, w].
    quaternion = state[QDIMS.quaternion_slice]
    return quaternion[[1, 2, 3, 0]]


def _motor_positions(arm_length: float) -> np.ndarray:
    return np.array(
        [
            [arm_length, 0.0, 0.0],
            [0.0, -arm_length, 0.0],
            [-arm_length, 0.0, 0.0],
            [0.0, arm_length, 0.0],
        ],
        dtype=np.float64,
    )


def _log_static_geometry(recording, rr, trajectory: Trajectory, arm_length: float) -> None:
    motor_positions = _motor_positions(arm_length)
    recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    recording.log(
        "world",
        rr.TransformAxes3D(axis_length=max(1.0, 2.0 * arm_length)),
        static=True,
    )
    recording.log(
        "world/trajectory",
        rr.LineStrips3D(
            [trajectory.states[:, QDIMS.position_slice]],
            radii=0.02,
            colors=[80, 160, 255],
        ),
        static=True,
    )
    recording.log(
        "world/quadrotor",
        rr.TransformAxes3D(axis_length=arm_length),
        static=True,
    )
    recording.log(
        "world/quadrotor/arms",
        rr.LineStrips3D(
            [
                [[arm_length, 0.0, 0.0], [-arm_length, 0.0, 0.0]],
                [[0.0, -arm_length, 0.0], [0.0, arm_length, 0.0]],
            ],
            radii=0.025,
            colors=[[255, 120, 80], [80, 200, 120]],
        ),
        static=True,
    )
    recording.log(
        "world/quadrotor/motors",
        rr.Points3D(
            motor_positions,
            radii=0.06,
            labels=[
                f"M{index}"
                for index in range(1, QDIMS.motor_count + 1)
            ],
            show_labels=True,
        ),
        static=True,
    )

    if trajectory.target_positions is not None and trajectory.target_positions.shape == (3,):
        recording.log(
            "world/target",
            rr.Points3D([trajectory.target_positions], radii=0.08, colors=[255, 180, 0]),
            static=True,
        )


def _log_pose(recording, rr, time: float, state: np.ndarray) -> None:
    recording.set_time("simulation_time", duration=float(time))
    recording.log(
        "world/quadrotor",
        rr.Transform3D(
            translation=state[0:3],
            quaternion=rr.Quaternion(xyzw=_quaternion_xyzw(state)),
        ),
    )


def _require_step_stride(step_stride: int) -> int:
    if isinstance(step_stride, bool) or not isinstance(step_stride, Integral):
        raise ValueError("step_stride must be a positive integer")
    if step_stride <= 0:
        raise ValueError("step_stride must be a positive integer")
    return int(step_stride)


def _sample_indices(count: int, step_stride: int) -> list[int]:
    if count == 0:
        return []
    indices = list(range(0, count, step_stride))
    # Keep the final sample so a sparse recording still shows the episode boundary.
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


def log_trajectory(
    trajectory: Trajectory,
    arm_length: float,
    recording_path: str | Path | None = None,
    application_id: str = "quadrotor_physics",
    spawn: bool = False,
    step_stride: int = 1,
) -> Path | None:
    """Log a sampled physical trajectory to Rerun."""
    if not isinstance(trajectory, Trajectory):
        raise TypeError("trajectory must be a Trajectory")
    _require_arm_length(arm_length)
    step_stride = _require_step_stride(step_stride)
    rr = _require_rerun()

    output_path = None if recording_path is None else Path(recording_path)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = _default_blueprint()
    with rr.RecordingStream(application_id) as recording:
        if output_path is not None:
            recording.save(output_path, default_blueprint=blueprint)
        if spawn:
            recording.spawn(default_blueprint=blueprint)
        _log_static_geometry(recording, rr, trajectory, arm_length)

        for index in _sample_indices(trajectory.times.shape[0], step_stride):
            _log_pose(recording, rr, trajectory.times[index], trajectory.states[index])
            if (
                trajectory.target_positions is not None
                and trajectory.target_positions.shape == (trajectory.times.shape[0], 3)
            ):
                recording.log(
                    "world/target",
                    rr.Points3D(
                        [trajectory.target_positions[index]],
                        radii=0.08,
                        colors=[255, 180, 0],
                    ),
                )

        for index in _sample_indices(trajectory.step_count, step_stride):
            recording.set_time(
                "simulation_time",
                duration=float(trajectory.times[index]),
            )
            for motor_index, thrust in enumerate(trajectory.thrusts[index], start=1):
                recording.log(
                    f"world/quadrotor/motors/M{motor_index}/thrust",
                    rr.Scalars([float(thrust)]),
                )
        recording.flush()
    return output_path
