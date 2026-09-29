import csv
from dataclasses import replace

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch

from research.pipeline.control.residual.models.experiments import load_experiment_config
from research.pipeline.control.rl.ppo import train_experiment


CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def test_training_loop_writes_run_logs_and_evaluation_artifacts(tmp_path) -> None:
    config = load_experiment_config(CONFIG_PATH)
    initializer = replace(
        config.environment.initializer,
        max_position_distance=0.0,
        max_velocity=0.0,
        max_tilt_angle=0.0,
        max_angular_velocity=0.0,
    )
    config = replace(
        config,
        environment=replace(
            config.environment,
            initializer=initializer,
            episode_horizon=0.02,
        ),
        ppo=replace(config.ppo, rollout_steps=4),
        evaluation=replace(config.evaluation, interval_updates=1, episodes=1),
        training=replace(config.training, updates=2),
    )

    run = train_experiment(config, runs_root=tmp_path)

    with run.metrics_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["update"] == "1"
    assert rows[0]["policy_loss"]
    assert rows[0]["policy_std_mean"]
    assert rows[0]["policy_std_min"]
    assert rows[0]["policy_std_max"]
    assert rows[0]["mean_raw_cost"]
    assert rows[0]["mean_normalized_cost"]
    assert rows[0]["evaluation_success_rate"]
    assert list(run.tensorboard_dir.glob("events.out.tfevents.*"))
    assert len(list(run.evaluation_dir.glob("*.npz"))) == 2
    assert len(list(run.checkpoints_dir.glob("step_*.pt"))) == 2
    assert (run.checkpoints_dir / "best.pt").exists()
    assert (run.checkpoints_dir / "final.pt").exists()

    checkpoint = torch.load(
        run.checkpoints_dir / "final.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["run_id"] == run.run_id
    assert checkpoint["config_hash"] == run.run_id.rsplit("_", maxsplit=1)[-1]
    assert checkpoint["config_path"] == str(run.config_path)
    assert checkpoint["resolved_config"] == config.as_dict()
    assert checkpoint["update"] == 2
    assert checkpoint["environment_steps"] == 8
    assert checkpoint["actor"]
    assert checkpoint["critic"]
    assert checkpoint["actor_optimizer"]
    assert checkpoint["critic_optimizer"]
    assert checkpoint["observation_normalizer"]["count"] > 0.0

    accumulator = EventAccumulator(str(run.tensorboard_dir))
    accumulator.Reload()
    assert {
        "ppo/policy_loss",
        "policy/std_mean",
        "reward/raw_cost",
        "reward/normalized_cost",
        "ppo/clip_fraction",
        "evaluation/success_rate",
    } <= set(accumulator.Tags()["scalars"])
