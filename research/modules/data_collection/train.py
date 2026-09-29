"""Run the configured PPO training experiment."""

import argparse
import sys
from pathlib import Path

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from research.pipeline.control.residual.models.experiments import load_experiment_config
from research.pipeline.control.rl.ppo import train_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ppo_hover_baseline.toml"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_experiment_config(args.config)
    run = train_experiment(
        config,
        runs_root=args.runs_root,
        device=args.device,
    )
    print(f"run: {run.run_dir}")
    print(f"metrics: {run.metrics_path}")
    print(f"tensorboard: {run.tensorboard_dir}")
    print(f"evaluation: {run.evaluation_dir}")


if __name__ == "__main__":
    main()
