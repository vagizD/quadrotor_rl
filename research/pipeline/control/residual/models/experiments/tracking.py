"""Persistent training metric logging."""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import TextIO

from torch.utils.tensorboard import SummaryWriter

TRAINING_METRIC_FIELDS = (
    "update",
    "environment_steps",
    "mean_episode_return",
    "rolling_episode_return",
    "mean_episode_length",
    "mean_raw_cost",
    "mean_normalized_cost",
    "rms_position_error",
    "policy_loss",
    "value_loss",
    "total_loss",
    "entropy",
    "mean_ratio",
    "clip_fraction",
    "approximate_kl",
    "explained_variance",
    "actor_grad_norm",
    "critic_grad_norm",
    "action_mean",
    "action_std",
    "policy_std_mean",
    "policy_std_min",
    "policy_std_max",
    "throughput_steps_per_second",
    "evaluation_rms_position_error",
    "evaluation_final_position_error",
    "evaluation_mean_speed",
    "evaluation_max_angular_rate",
    "evaluation_success_rate",
    "evaluation_failure_rate",
    "evaluation_truncation_rate",
)

TENSORBOARD_METRIC_TAGS = {
    "update": "training/update",
    "environment_steps": "training/environment_steps",
    "mean_episode_return": "episode/mean_return",
    "rolling_episode_return": "episode/rolling_return",
    "mean_episode_length": "episode/mean_length",
    "mean_raw_cost": "reward/raw_cost",
    "mean_normalized_cost": "reward/normalized_cost",
    "rms_position_error": "physics/rms_position_error",
    "policy_loss": "ppo/policy_loss",
    "value_loss": "ppo/value_loss",
    "total_loss": "ppo/total_loss",
    "entropy": "ppo/entropy",
    "mean_ratio": "ppo/mean_ratio",
    "clip_fraction": "ppo/clip_fraction",
    "approximate_kl": "ppo/approximate_kl",
    "explained_variance": "value/explained_variance",
    "actor_grad_norm": "optimization/actor_grad_norm",
    "critic_grad_norm": "optimization/critic_grad_norm",
    "action_mean": "action/mean",
    "action_std": "action/std",
    "policy_std_mean": "policy/std_mean",
    "policy_std_min": "policy/std_min",
    "policy_std_max": "policy/std_max",
    "throughput_steps_per_second": "performance/steps_per_second",
    "success_rate": "episode/success_rate",
    "evaluation_rms_position_error": "evaluation/rms_position_error",
    "evaluation_final_position_error": "evaluation/final_position_error",
    "evaluation_mean_speed": "evaluation/mean_speed",
    "evaluation_max_angular_rate": "evaluation/max_angular_rate",
    "evaluation_success_rate": "evaluation/success_rate",
    "evaluation_failure_rate": "evaluation/failure_rate",
    "evaluation_truncation_rate": "evaluation/truncation_rate",
}


class CSVMetricLogger:
    """Append validated rows to a run-local metrics CSV file."""

    def __init__(
        self,
        path: str | Path,
        fieldnames: Sequence[str] = TRAINING_METRIC_FIELDS,
    ) -> None:
        fieldnames = tuple(fieldnames)
        if not fieldnames or len(set(fieldnames)) != len(fieldnames):
            raise ValueError("fieldnames must be a non-empty unique sequence")
        if any(not isinstance(field, str) or not field for field in fieldnames):
            raise ValueError("fieldnames must contain non-empty strings")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._file: TextIO = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, metrics: Mapping[str, Real]) -> None:
        unknown = set(metrics) - set(self.fieldnames)
        if unknown:
            raise ValueError(f"unknown metric fields: {sorted(unknown)}")
        row: dict[str, Real | str] = {}
        for field in self.fieldnames:
            value = metrics.get(field, "")
            if value != "":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"metric {field} must be finite and numeric")
            row[field] = value
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CSVMetricLogger":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()


class ConsoleProgressReporter:
    """Print compact training progress at a fixed update interval."""

    def __init__(
        self,
        interval_updates: int = 1,
        stream: TextIO | None = None,
    ) -> None:
        if (
            isinstance(interval_updates, bool)
            or not isinstance(interval_updates, Integral)
            or interval_updates <= 0
        ):
            raise ValueError("interval_updates must be a positive integer")
        self.interval_updates = int(interval_updates)
        self.stream = sys.stdout if stream is None else stream

    def report(self, metrics: Mapping[str, Real]) -> bool:
        """Print one row when its update reaches the configured interval."""
        unknown = set(metrics) - set(TRAINING_METRIC_FIELDS)
        if unknown:
            raise ValueError(f"unknown metric fields: {sorted(unknown)}")
        update = metrics.get("update")
        if (
            isinstance(update, bool)
            or not isinstance(update, Real)
            or not math.isfinite(float(update))
            or int(update) != update
            or update < 0
        ):
            raise ValueError("metric update must be a non-negative integer")
        update = int(update)
        if update % self.interval_updates != 0:
            return False

        values = [
            f"{field}={_format_metric(metrics.get(field))}"
            for field in TRAINING_METRIC_FIELDS
        ]
        print(" | ".join(values), file=self.stream, flush=True)
        return True


def _format_metric(value: Real | None) -> str:
    if value is None:
        return "-"
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError("metrics must be finite and numeric")
    if isinstance(value, Integral):
        return str(int(value))
    return f"{float(value):.6g}"


class TensorBoardMetricLogger:
    """Write training scalars to a run-local TensorBoard event stream."""

    def __init__(self, log_dir: str | Path, flush_secs: float = 1.0) -> None:
        if not math.isfinite(flush_secs) or flush_secs <= 0.0:
            raise ValueError("flush_secs must be finite and positive")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_secs,
        )

    def log(self, metrics: Mapping[str, Real]) -> None:
        unknown = set(metrics) - set(TENSORBOARD_METRIC_TAGS)
        if unknown:
            raise ValueError(f"unknown metric fields: {sorted(unknown)}")
        update = metrics.get("update")
        if (
            isinstance(update, bool)
            or not isinstance(update, Real)
            or not math.isfinite(float(update))
            or int(update) != update
            or update < 0
        ):
            raise ValueError("metric update must be a non-negative integer")
        for field, value in metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"metric {field} must be finite and numeric")
            self._writer.add_scalar(
                TENSORBOARD_METRIC_TAGS[field],
                float(value),
                global_step=int(update),
            )
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "TensorBoardMetricLogger":
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()
