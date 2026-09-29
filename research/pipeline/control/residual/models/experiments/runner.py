"""Experiment run-directory creation and metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import ExperimentConfig, save_resolved_config


@dataclass(frozen=True)
class RunDirectory:
    run_id: str
    run_dir: Path
    config_path: Path
    metadata_path: Path
    metrics_path: Path
    tensorboard_dir: Path
    checkpoints_dir: Path
    plots_dir: Path
    evaluation_dir: Path


def create_run_directory(
    config: ExperimentConfig,
    runs_root: str | Path = "runs",
    device: str = "cpu",
    start_time: datetime | None = None,
    git_commit: str | None = None,
) -> RunDirectory:
    """Create a run directory and freeze its initial metadata."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")

    timestamp = datetime.now(timezone.utc) if start_time is None else start_time
    if timestamp.tzinfo is None:
        raise ValueError("start_time must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    config_hash = _config_hash(config)
    run_id = f"{timestamp:%Y%m%d_%H%M%S}_seed{config.seed}_{config_hash}"
    run_dir = Path(runs_root) / config.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config.toml"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.csv"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoints_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    evaluation_dir = run_dir / "evaluation"
    for directory in (
        checkpoints_dir,
        plots_dir,
        evaluation_dir,
        tensorboard_dir,
    ):
        directory.mkdir()

    save_resolved_config(config, config_path)
    metadata = {
        "experiment_name": config.name,
        "seed": config.seed,
        "config_hash": config_hash,
        "git_commit": git_commit if git_commit is not None else _git_commit(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "start_time": timestamp.isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RunDirectory(
        run_id=run_id,
        run_dir=run_dir,
        config_path=config_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        tensorboard_dir=tensorboard_dir,
        checkpoints_dir=checkpoints_dir,
        plots_dir=plots_dir,
        evaluation_dir=evaluation_dir,
    )


def _config_hash(config: ExperimentConfig) -> str:
    serialized = json.dumps(
        config.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:8]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None
