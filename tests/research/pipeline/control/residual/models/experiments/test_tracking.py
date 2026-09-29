import csv
from io import StringIO

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from research.pipeline.control.residual.models.experiments import (
    CSVMetricLogger,
    ConsoleProgressReporter,
    TensorBoardMetricLogger,
    TRAINING_METRIC_FIELDS,
)


def test_csv_metric_logger_writes_core_training_rows(tmp_path) -> None:
    path = tmp_path / "metrics.csv"

    with CSVMetricLogger(path) as logger:
        logger.log(
            {
                "update": 1,
                "environment_steps": 16,
                "mean_episode_return": 3.5,
                "policy_loss": -0.2,
            }
        )
        logger.log(
            {
                "update": 2,
                "environment_steps": 32,
                "mean_episode_return": 4.0,
                "policy_loss": -0.1,
            }
        )

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0].keys() == set(TRAINING_METRIC_FIELDS)
    assert rows[0]["update"] == "1"
    assert rows[0]["environment_steps"] == "16"
    assert rows[0]["mean_episode_return"] == "3.5"
    assert rows[0]["value_loss"] == ""
    assert rows[1]["update"] == "2"
    assert rows[1]["mean_episode_return"] == "4.0"


def test_console_progress_reporter_respects_update_interval() -> None:
    stream = StringIO()
    reporter = ConsoleProgressReporter(interval_updates=2, stream=stream)

    assert not reporter.report({"update": 1})
    assert reporter.report(
        {
            "update": 2,
            "environment_steps": 64,
            "mean_episode_return": 4.0,
            "rolling_episode_return": 3.5,
            "mean_episode_length": 8,
            "rms_position_error": 0.2,
            "policy_loss": -0.1,
            "value_loss": 0.4,
            "throughput_steps_per_second": 1200.0,
        }
    )

    output = stream.getvalue()
    assert "update=2" in output
    assert "environment_steps=64" in output
    assert "rolling_episode_return=3.5" in output
    assert "throughput_steps_per_second=1200" in output


def test_tensorboard_metric_logger_writes_scalar_events(tmp_path) -> None:
    log_dir = tmp_path / "tensorboard"

    with TensorBoardMetricLogger(log_dir) as logger:
        logger.log(
            {
                "update": 3,
                "environment_steps": 96,
                "mean_episode_return": 4.0,
                "rolling_episode_return": 3.5,
                "policy_loss": -0.1,
                "action_mean": 2.0,
                "policy_std_mean": 0.25,
                "mean_raw_cost": 1.0,
                "mean_normalized_cost": 0.15,
                "evaluation_rms_position_error": 0.15,
                "evaluation_success_rate": 0.5,
            }
        )

    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert event_files
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    assert {
        "training/environment_steps",
        "episode/mean_return",
        "episode/rolling_return",
        "ppo/policy_loss",
        "action/mean",
        "policy/std_mean",
        "reward/raw_cost",
        "reward/normalized_cost",
        "evaluation/rms_position_error",
        "evaluation/success_rate",
    } <= set(accumulator.Tags()["scalars"])
    event = accumulator.Scalars("episode/mean_return")[0]
    assert event.step == 3
    assert event.value == 4.0
