"""Static plots for recorded quadrotor trajectories."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from geometry.common.dimensions import QDIMS
from research.pipeline.planning.trajectory import Trajectory

TimeAxis = Literal["time", "step", "both"]


def _x_values(
    sample_times: np.ndarray,
    time_axis: TimeAxis,
) -> tuple[np.ndarray, str]:
    if time_axis == "time" or time_axis == "both":
        return sample_times, "time [s]"
    if time_axis == "step":
        return np.arange(sample_times.size, dtype=np.float64), "step index"
    raise ValueError("time_axis must be 'time', 'step', or 'both'")


def _add_step_axis(axis, trajectory_times: np.ndarray) -> None:
    steps = np.arange(trajectory_times.size, dtype=np.float64)
    time_to_step = lambda values: np.interp(values, trajectory_times, steps)
    step_to_time = lambda values: np.interp(values, steps, trajectory_times)
    secondary = axis.secondary_xaxis(
        "top",
        functions=(time_to_step, step_to_time),
    )
    secondary.set_xlabel("step index")


def _add_components(
    axes: np.ndarray,
    x_values: np.ndarray,
    values: np.ndarray,
    labels: tuple[str, ...],
    ylabel: str,
    reference: np.ndarray | None = None,
    x_label: str = "time [s]",
) -> None:
    for index, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(x_values, values[:, index], label=label)
        if reference is not None:
            axis.plot(x_values, reference[:, index], "--", label=f"target {label}")
        axis.set_ylabel(f"{label} {ylabel}" if ylabel else label)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel(x_label)


def _component_figure(
    times: np.ndarray,
    values: np.ndarray,
    labels: tuple[str, ...],
    title: str,
    ylabel: str,
    reference: np.ndarray | None = None,
    time_axis: TimeAxis = "time",
    trajectory_times: np.ndarray | None = None,
) -> Figure:
    figure, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(8.0, 2.0 * len(labels)))
    axes = np.atleast_1d(axes)
    x_values, x_label = _x_values(times, time_axis)
    _add_components(axes, x_values, values, labels, ylabel, reference, x_label)
    if time_axis == "both" and trajectory_times is not None:
        _add_step_axis(axes[-1], trajectory_times)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def _position_targets(trajectory: Trajectory) -> np.ndarray | None:
    if trajectory.target_positions is None:
        return None
    return np.broadcast_to(
        trajectory.target_positions,
        trajectory.states.shape[0:1] + (3,),
    )


def plot_position(trajectory: Trajectory, time_axis: TimeAxis = "time") -> Figure:
    return _component_figure(
        trajectory.times,
        trajectory.states[:, QDIMS.position_slice],
        ("x", "y", "z"),
        "Position",
        "[m]",
        _position_targets(trajectory),
        time_axis,
        trajectory.times,
    )


def plot_linear_velocity(trajectory: Trajectory, time_axis: TimeAxis = "time") -> Figure:
    return _component_figure(
        trajectory.times,
        trajectory.states[:, QDIMS.velocity_slice],
        ("vx", "vy", "vz"),
        "Linear velocity",
        "[m/s]",
        time_axis=time_axis,
        trajectory_times=trajectory.times,
    )


def plot_angular_velocity(trajectory: Trajectory, time_axis: TimeAxis = "time") -> Figure:
    return _component_figure(
        trajectory.times,
        trajectory.states[:, QDIMS.angular_velocity_slice],
        ("wx", "wy", "wz"),
        "Angular velocity",
        "[rad/s]",
        time_axis=time_axis,
        trajectory_times=trajectory.times,
    )


def plot_quaternion(trajectory: Trajectory, time_axis: TimeAxis = "time") -> Figure:
    return _component_figure(
        trajectory.times,
        trajectory.states[:, QDIMS.quaternion_slice],
        ("qw", "qx", "qy", "qz"),
        "Quaternion",
        "",
        time_axis=time_axis,
        trajectory_times=trajectory.times,
    )


def plot_thrusts(trajectory: Trajectory, time_axis: TimeAxis = "time") -> Figure:
    return _component_figure(
        trajectory.times[:-1],
        trajectory.thrusts,
        tuple(
            f"f{index}" for index in range(1, QDIMS.motor_count + 1)
        ),
        "Motor thrusts",
        "[N]",
        time_axis=time_axis,
        trajectory_times=trajectory.times,
    )


def plot_trajectory(
    trajectory: Trajectory,
    include_quaternion: bool = True,
    time_axis: TimeAxis = "time",
) -> Figure:
    row_count = QDIMS.state_dim + QDIMS.motor_count * int(
        include_quaternion
    )
    figure, axes = plt.subplots(
        row_count,
        1,
        sharex=True,
        figsize=(9.0, 2.0 * row_count),
    )
    axes = np.atleast_1d(axes)
    position_targets = _position_targets(trajectory)
    state_x, x_label = _x_values(trajectory.times, time_axis)
    thrust_x, _ = _x_values(trajectory.times[:-1], time_axis)
    cursor = 0
    _add_components(axes[cursor:cursor + QDIMS.position_dim], state_x, trajectory.states[:, QDIMS.position_slice], ("x", "y", "z"), "[m]", position_targets, x_label)
    cursor += QDIMS.position_dim
    _add_components(axes[cursor:cursor + QDIMS.velocity_dim], state_x, trajectory.states[:, QDIMS.velocity_slice], ("vx", "vy", "vz"), "[m/s]", x_label=x_label)
    cursor += QDIMS.velocity_dim
    if include_quaternion:
        _add_components(axes[cursor:cursor + QDIMS.quaternion_dim], state_x, trajectory.states[:, QDIMS.quaternion_slice], ("qw", "qx", "qy", "qz"), "", x_label=x_label)
        cursor += QDIMS.quaternion_dim
    _add_components(axes[cursor:cursor + QDIMS.angular_velocity_dim], state_x, trajectory.states[:, QDIMS.angular_velocity_slice], ("wx", "wy", "wz"), "[rad/s]", x_label=x_label)
    cursor += QDIMS.angular_velocity_dim
    _add_components(axes[cursor:cursor + QDIMS.motor_count], thrust_x, trajectory.thrusts, tuple(f"f{index}" for index in range(1, QDIMS.motor_count + 1)), "[N]", x_label=x_label)
    if time_axis == "both":
        _add_step_axis(axes[-1], trajectory.times)
    figure.suptitle("Quadrotor trajectory")
    figure.tight_layout()
    return figure


def save_trajectory_plot(
    trajectory: Trajectory,
    path: str | Path,
    include_quaternion: bool = True,
    time_axis: TimeAxis = "time",
) -> Path:
    figure = plot_trajectory(trajectory, include_quaternion, time_axis)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
