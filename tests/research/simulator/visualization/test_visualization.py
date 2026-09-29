import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry.common.dimensions import QDIMS
from research.simulator.integrators import EulerIntegrator, Trajectory
from robots.quadrotor.recorder import record_quadrotor_trajectory
from tests.common import make_quadrotor
from research.simulator.visualization.plotting import (
    plot_angular_velocity,
    plot_linear_velocity,
    plot_position,
    plot_quaternion,
    plot_thrusts,
    plot_trajectory,
    save_trajectory_plot,
)


def make_recorded_trajectory() -> Trajectory:
    quadrotor = make_quadrotor()
    hover_thrust = (
        quadrotor.params.mass
        * quadrotor.params.gravity
        / QDIMS.motor_count
    )
    thrust_schedule = np.full((3, QDIMS.motor_count), hover_thrust)
    return record_quadrotor_trajectory(
        quadrotor,
        EulerIntegrator(dt=0.01),
        thrust_schedule,
        target_positions=np.array([0.0, 0.0, 1.0]),
    )


def test_component_plots_consume_generated_trajectory() -> None:
    trajectory = make_recorded_trajectory()

    figures = [
        plot_position(trajectory),
        plot_linear_velocity(trajectory),
        plot_angular_velocity(trajectory),
        plot_quaternion(trajectory),
        plot_thrusts(trajectory),
    ]

    assert [len(figure.axes) for figure in figures] == [
        QDIMS.position_dim,
        QDIMS.velocity_dim,
        QDIMS.angular_velocity_dim,
        QDIMS.quaternion_dim,
        QDIMS.motor_count,
    ]
    assert all(len(axis.lines) == 2 for axis in figures[0].axes)
    for figure in figures:
        plt.close(figure)


def test_combined_plot_can_omit_quaternion_components() -> None:
    trajectory = make_recorded_trajectory()

    with_quaternion = plot_trajectory(trajectory)
    without_quaternion = plot_trajectory(trajectory, include_quaternion=False)

    assert len(with_quaternion.axes) == 17
    assert len(without_quaternion.axes) == (
        QDIMS.state_dim
        - QDIMS.quaternion_dim
        + QDIMS.motor_count
    )
    plt.close(with_quaternion)
    plt.close(without_quaternion)


def test_plots_support_step_or_time_on_the_horizontal_axis() -> None:
    trajectory = make_recorded_trajectory()

    step_figure = plot_position(trajectory, time_axis="step")
    both_figure = plot_position(trajectory, time_axis="both")

    assert step_figure.axes[-1].get_xlabel() == "step index"
    assert both_figure.axes[-1].get_xlabel() == "time [s]"
    assert both_figure.axes[-1].secondary_xaxis("top") is not None
    plt.close(step_figure)
    plt.close(both_figure)


def test_trajectory_recording_and_plot_can_be_saved(tmp_path) -> None:
    trajectory = make_recorded_trajectory()
    recording_path = tmp_path / "hover.npz"
    no_target_path = tmp_path / "free_fall.npz"
    plot_path = tmp_path / "plots" / "hover.png"

    trajectory.save(recording_path)
    Trajectory(
        times=trajectory.times,
        states=trajectory.states,
        thrusts=trajectory.thrusts,
    ).save(no_target_path)
    saved_plot = save_trajectory_plot(trajectory, plot_path)
    restored = Trajectory.load(recording_path)
    restored_without_target = Trajectory.load(no_target_path)

    assert saved_plot == plot_path
    assert plot_path.exists() and plot_path.stat().st_size > 0
    np.testing.assert_allclose(restored.times, trajectory.times)
    np.testing.assert_allclose(restored.states, trajectory.states)
    np.testing.assert_allclose(restored.thrusts, trajectory.thrusts)
    np.testing.assert_allclose(restored.target_positions, trajectory.target_positions)
    assert restored_without_target.target_positions is None
