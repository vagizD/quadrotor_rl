"""Generate and visualize deterministic quadrotor physics diagnostics."""

import argparse
import sys
from pathlib import Path

import numpy as np

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from geometry.rotation import Quaternion
from common.dimensions import QDIMS
from robots.quadrotor import Quadrotor, QuadrotorParams, QuadrotorState
from simulation import EulerIntegrator, RK4Integrator, record_quadrotor_trajectory
from visualization.plotting import save_trajectory_plot


SCENARIOS = ("free_fall", "hover", "climb", "roll", "pitch", "yaw", "tilted_thrust")
INTEGRATORS = {"euler": EulerIntegrator, "rk4": RK4Integrator}


def _make_quadrotor(quaternion: Quaternion | None = None) -> Quadrotor:
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


def _scenario_data(
    scenario: str,
    steps: int,
) -> tuple[Quadrotor, np.ndarray, np.ndarray | None]:
    if scenario == "tilted_thrust":
        angle = np.deg2rad(20.0)
        quadrotor = _make_quadrotor(
            Quaternion(np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0)
        )
    else:
        quadrotor = _make_quadrotor()

    hover_thrust = (
        quadrotor.params.mass
        * quadrotor.params.gravity
        / QDIMS.motor_count
    )
    target = None
    if scenario == "free_fall":
        thrusts = np.zeros(QDIMS.motor_count)
    elif scenario == "hover":
        thrusts = np.full(QDIMS.motor_count, hover_thrust)
        target = np.zeros(3)
    elif scenario == "climb":
        thrusts = np.full(QDIMS.motor_count, hover_thrust + 0.25)
        target = np.array([0.0, 0.0, 1.0])
    elif scenario == "roll":
        thrusts = np.array([2.0, 1.0, 2.0, 3.0])
    elif scenario == "pitch":
        thrusts = np.array([1.0, 2.0, 3.0, 2.0])
    elif scenario == "yaw":
        thrusts = np.array([2.0, 1.0, 2.0, 1.0])
    elif scenario == "tilted_thrust":
        thrusts = np.full(QDIMS.motor_count, hover_thrust)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return quadrotor, np.tile(thrusts, (steps, 1)), target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="hover")
    parser.add_argument("--integrator", choices=tuple(INTEGRATORS), default="euler")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/physics_diagnostics"))
    parser.add_argument("--time-axis", choices=("time", "step", "both"), default="time")
    parser.add_argument("--rerun", action="store_true", help="also save a Rerun .rrd recording")
    parser.add_argument("--spawn", action="store_true", help="spawn the Rerun viewer")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.dt <= 0.0 or not np.isfinite(args.dt):
        parser.error("--dt must be finite and positive")
    if args.duration <= 0.0 or not np.isfinite(args.duration):
        parser.error("--duration must be finite and positive")
    if args.spawn and not args.rerun:
        parser.error("--spawn requires --rerun")

    steps = max(1, int(round(args.duration / args.dt)))
    quadrotor, thrust_schedule, target = _scenario_data(args.scenario, steps)
    integrator = INTEGRATORS[args.integrator](dt=args.dt)
    trajectory = record_quadrotor_trajectory(
        quadrotor,
        integrator,
        thrust_schedule,
        target,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}_{args.integrator}"
    trajectory_path = args.output_dir / f"{stem}.npz"
    plot_path = args.output_dir / f"{stem}.png"
    trajectory.save(trajectory_path)
    save_trajectory_plot(trajectory, plot_path, time_axis=args.time_axis)
    print(f"trajectory: {trajectory_path}")
    print(f"plot: {plot_path}")

    if args.rerun:
        from visualization.rerun_viewer import log_trajectory

        rerun_path = args.output_dir / f"{stem}.rrd"
        log_trajectory(
            trajectory,
            quadrotor.params.arm_length,
            recording_path=rerun_path,
            application_id=f"quadrotor_physics_{args.scenario}",
            spawn=args.spawn,
        )
        print(f"rerun: {rerun_path}")


if __name__ == "__main__":
    main()
