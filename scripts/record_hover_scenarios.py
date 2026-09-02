"""Record deterministic hover scenarios for a trained PPO run."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from environments import HoverEnvironment, get_hover_scenarios
from experiments import load_experiment_config
from rl.ppo import Actor, DeterministicEvaluator, ObservationNormalizer
from visualization.rerun_viewer import log_trajectory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="run identifier under runs/<experiment>/")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--checkpoint",
        choices=("final", "best"),
        default="final",
        help="checkpoint to evaluate (default: final)",
    )
    parser.add_argument("--device", default="cpu")
    return parser


def _find_run_dir(runs_root: Path, run_id: str) -> Path:
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be a directory name, not a path")
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {runs_root}")
    candidates = [
        experiment_dir / run_id
        for experiment_dir in runs_root.iterdir()
        if experiment_dir.is_dir() and (experiment_dir / run_id).is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"could not find run_id {run_id!r} below {runs_root}"
        )
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path in candidates)
        raise ValueError(f"run_id is ambiguous; matching directories: {paths}")
    return candidates[0]


def _load_checkpoint(
    run_dir: Path,
    run_id: str,
    checkpoint_name: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoints" / f"{checkpoint_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")
    if checkpoint.get("run_id") != run_id:
        raise ValueError("checkpoint run_id does not match the requested run")
    if "actor" not in checkpoint:
        raise ValueError("checkpoint does not contain actor parameters")
    return checkpoint


def _make_actor(config, checkpoint: dict[str, Any], device: torch.device) -> Actor:
    actor = Actor(
        config.dimensions.observation_dim,
        config.dimensions.action_dim,
        config.ppo.hidden_sizes,
        config.ppo.initial_log_std,
        config.ppo.min_log_std,
        config.ppo.max_log_std,
        gated_dual_policy=config.ppo.gated_dual_policy,
        transit_hidden_sizes=config.ppo.transit_hidden_sizes,
    ).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def _make_observation_normalizer(
    config,
    checkpoint: dict[str, Any],
) -> ObservationNormalizer | None:
    state = checkpoint.get("observation_normalizer")
    if state is None:
        return None
    if not isinstance(state, dict):
        raise ValueError("checkpoint observation_normalizer must be a mapping")
    normalizer = ObservationNormalizer(
        config.dimensions.observation_dim,
        config.ppo.observation_clip,
        config.ppo.normalization_epsilon,
    )
    normalizer.mean = np.array(state["mean"], dtype=np.float64, copy=True)
    normalizer.variance = np.array(
        state["variance"],
        dtype=np.float64,
        copy=True,
    )
    normalizer.count = float(state["count"])
    return normalizer


def _record_scenario(
    environment: HoverEnvironment,
    actor: Actor,
    config,
    normalizer: ObservationNormalizer | None,
    scenario,
    output_dir: Path,
) -> None:
    initial_state = scenario.initial_state(
        environment.target_position,
        environment.target_heading,
    )
    evaluator = DeterministicEvaluator(
        environment,
        actor,
        replace(config.evaluation, episodes=1),
        initial_states=(initial_state,),
        observation_normalizer=normalizer,
    )
    summary = evaluator.evaluate()
    episode = summary.episodes[0]
    trajectory_path = output_dir / f"{scenario.name}.npz"
    rerun_path = output_dir / f"{scenario.name}.rrd"
    episode.trajectory.save(trajectory_path)
    log_trajectory(
        episode.trajectory,
        environment.params.arm_length,
        recording_path=rerun_path,
        application_id=f"{config.name}_hover_scenarios",
        step_stride=config.tracking.rerun_step_stride,
    )
    print(
        f"{scenario.name}: "
        f"final_position_error={episode.final_position_error:.6f}, "
        f"rms_position_error={episode.rms_position_error:.6f}, "
        f"success={episode.success}, "
        f"terminated={episode.terminated}, "
        f"truncated={episode.truncated}"
    )
    print(f"  trajectory: {trajectory_path}")
    print(f"  rerun: {rerun_path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        device = torch.device(args.device)
        run_dir = _find_run_dir(args.runs_root, args.run_id)
        config = load_experiment_config(run_dir / "config.toml")
        checkpoint = _load_checkpoint(
            run_dir,
            args.run_id,
            args.checkpoint,
            device,
        )
        actor = _make_actor(config, checkpoint, device)
        normalizer = _make_observation_normalizer(config, checkpoint)
        environment = HoverEnvironment.from_config(config)
        output_dir = run_dir / "evaluation" / "hover_scenarios"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"run: {run_dir}")
        print(f"checkpoint: {args.checkpoint}")
        for scenario in get_hover_scenarios():
            _record_scenario(
                environment,
                actor,
                config,
                normalizer,
                scenario,
                output_dir,
            )
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
