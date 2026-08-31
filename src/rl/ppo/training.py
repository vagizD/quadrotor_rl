"""Repeated PPO updates with run-local monitoring."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from common.dimensions import QDIMS
from experiments import (
    CSVMetricLogger,
    ConsoleProgressReporter,
    ExperimentConfig,
    RunDirectory,
    TensorBoardMetricLogger,
    create_run_directory,
)
from environments import HoverEnvironment

from .evaluation import (
    DeterministicEvaluator,
    EvaluationEpisode,
    EvaluationSummary,
)
from .policy import Actor, Critic
from .normalization import ObservationNormalizer
from .rollout import PPORollout, collect_ppo_rollout
from .trainer import PPOTrainer


def train_experiment(
    config: ExperimentConfig,
    runs_root: str | Path = "runs",
    device: str = "cpu",
) -> RunDirectory:
    """Run the configured PPO updates and return the created run directory."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    torch_device = torch.device(device)
    torch.manual_seed(config.seed)
    environment = HoverEnvironment.from_config(config)
    actor = Actor(
        config.dimensions.observation_dim,
        config.dimensions.action_dim,
        config.ppo.hidden_sizes,
        config.ppo.initial_log_std,
        config.ppo.min_log_std,
        config.ppo.max_log_std,
    ).to(torch_device)
    critic = Critic(
        config.dimensions.observation_dim,
        config.ppo.hidden_sizes,
    ).to(torch_device)
    trainer = PPOTrainer(
        actor,
        critic,
        config.ppo,
        action_low=0.0,
        action_high=config.quadrotor.max_thrust,
    )
    observation_normalizer = (
        ObservationNormalizer(
            config.dimensions.observation_dim,
            config.ppo.observation_clip,
            config.ppo.normalization_epsilon,
        )
        if config.ppo.normalize_observations
        else None
    )
    evaluator = DeterministicEvaluator(
        environment,
        actor,
        config.evaluation,
        observation_normalizer=observation_normalizer,
    )
    run = create_run_directory(config, runs_root=runs_root, device=device)
    rolling_returns: deque[float] = deque(
        maxlen=config.training.rolling_episode_window
    )
    environment_steps = 0
    best_evaluation_score: tuple[float, ...] | None = None
    last_checkpoint: dict[str, Any] | None = None

    with (
        CSVMetricLogger(run.metrics_path) as csv_logger,
        TensorBoardMetricLogger(run.tensorboard_dir) as tensorboard_logger,
    ):
        reporter = ConsoleProgressReporter(
            config.tracking.console_log_interval
        )
        for update in range(1, config.training.updates + 1):
            start = perf_counter()
            rollout = collect_ppo_rollout(
                environment,
                actor,
                critic,
                config.ppo.rollout_steps,
                observation_normalizer=observation_normalizer,
            )
            update_metrics = trainer.update(rollout)
            elapsed = perf_counter() - start
            environment_steps += rollout.step_count
            metrics = _training_metrics(
                update,
                environment_steps,
                rollout,
                update_metrics,
                elapsed,
                rolling_returns,
                actor,
            )
            if evaluator.should_evaluate(update):
                summary = evaluator.evaluate()
                metrics.update(summary.as_metrics(update))
                _save_evaluation(
                    run.evaluation_dir,
                    config,
                    update,
                    summary.episodes,
                )
                score = _evaluation_score(summary)
                if best_evaluation_score is None or score > best_evaluation_score:
                    best_evaluation_score = score
                    checkpoint = _checkpoint_payload(
                        config,
                        run,
                        actor,
                        critic,
                        trainer,
                        update,
                        environment_steps,
                        best_evaluation_score,
                        observation_normalizer,
                    )
                    _save_checkpoint(run.checkpoints_dir / "best.pt", checkpoint)
            csv_logger.log(metrics)
            tensorboard_logger.log(metrics)
            reporter.report(metrics)
            last_checkpoint = _checkpoint_payload(
                config,
                run,
                actor,
                critic,
                trainer,
                update,
                environment_steps,
                best_evaluation_score,
                observation_normalizer,
            )
            _save_checkpoint(
                run.checkpoints_dir / f"step_{update:06d}.pt",
                last_checkpoint,
            )
    if last_checkpoint is None:
        raise RuntimeError("training produced no checkpoint")
    _save_checkpoint(run.checkpoints_dir / "final.pt", last_checkpoint)
    return run


def _training_metrics(
    update: int,
    environment_steps: int,
    rollout: PPORollout,
    update_metrics,
    elapsed: float,
    rolling_returns: deque[float],
    actor: Actor,
) -> dict[str, float | int]:
    complete = [
        trajectory
        for trajectory in rollout.trajectories
        if trajectory.terminated[-1] or trajectory.truncated[-1]
    ]
    episode_returns = [float(np.sum(trajectory.rewards)) for trajectory in complete]
    rolling_returns.extend(episode_returns)
    raw_costs = [
        trajectory.raw_costs
        for trajectory in rollout.trajectories
        if trajectory.raw_costs is not None
    ]
    normalized_costs = [
        trajectory.normalized_costs
        for trajectory in rollout.trajectories
        if trajectory.normalized_costs is not None
    ]
    metrics: dict[str, float | int] = {
        "update": update,
        "environment_steps": environment_steps,
        **asdict(update_metrics),
        "throughput_steps_per_second": rollout.step_count / max(elapsed, 1e-12),
        "action_mean": float(
            np.mean(
                np.concatenate(
                    [trajectory.actions for trajectory in rollout.trajectories],
                    axis=0,
                )
            )
        ),
        "action_std": float(
            np.std(
                np.concatenate(
                    [trajectory.actions for trajectory in rollout.trajectories],
                    axis=0,
                )
            )
        ),
        "policy_std_mean": float(actor.policy_std.detach().mean().item()),
        "policy_std_min": float(actor.policy_std.detach().min().item()),
        "policy_std_max": float(actor.policy_std.detach().max().item()),
    }
    if raw_costs and len(raw_costs) == len(normalized_costs):
        raw_cost_values = np.concatenate(raw_costs)
        normalized_cost_values = np.concatenate(normalized_costs)
        metrics.update(
            {
                "mean_raw_cost": float(np.mean(raw_cost_values)),
                "mean_normalized_cost": float(np.mean(normalized_cost_values)),
            }
        )
    if episode_returns:
        metrics["mean_episode_return"] = float(np.mean(episode_returns))
        metrics["rolling_episode_return"] = float(np.mean(rolling_returns))
        metrics["mean_episode_length"] = float(
            np.mean([trajectory.step_count for trajectory in complete])
        )
    position_errors = []
    for trajectory in rollout.trajectories:
        if trajectory.target_positions is None:
            continue
        target = trajectory.target_positions
        if target.shape != (trajectory.times.shape[0], 3):
            target = np.broadcast_to(target, (trajectory.times.shape[0], 3))
        position_errors.append(
            np.linalg.norm(
                trajectory.states[:, QDIMS.position_slice] - target,
                axis=1,
            )
        )
    if position_errors:
        metrics["rms_position_error"] = float(
            np.sqrt(np.mean(np.concatenate(position_errors) ** 2))
        )
    return metrics


def _save_evaluation(
    evaluation_dir: Path,
    config: ExperimentConfig,
    update: int,
    episodes: tuple[EvaluationEpisode, ...],
) -> None:
    for episode_index, episode in enumerate(episodes, start=1):
        stem = f"update_{update:06d}_episode_{episode_index:03d}"
        trajectory_path = evaluation_dir / f"{stem}.npz"
        episode.trajectory.save(trajectory_path)
        if config.tracking.record_evaluation_rerun:
            from visualization.rerun_viewer import log_trajectory

            log_trajectory(
                episode.trajectory,
                config.quadrotor.arm_length,
                recording_path=evaluation_dir / f"{stem}.rrd",
                application_id=f"{config.name}_evaluation",
                step_stride=config.tracking.rerun_step_stride,
            )


def _evaluation_score(summary: EvaluationSummary) -> tuple[float, ...]:
    metrics = summary.as_metrics(update=0)
    return (
        float(metrics["evaluation_success_rate"]),
        -float(metrics["evaluation_rms_position_error"]),
        -float(metrics["evaluation_final_position_error"]),
        -float(metrics["evaluation_mean_speed"]),
        -float(metrics["evaluation_max_angular_rate"]),
    )


def _checkpoint_payload(
    config: ExperimentConfig,
    run: RunDirectory,
    actor: Actor,
    critic: Critic,
    trainer: PPOTrainer,
    update: int,
    environment_steps: int,
    best_evaluation_score: tuple[float, ...] | None,
    observation_normalizer: ObservationNormalizer | None,
) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "run_id": run.run_id,
        "config_hash": run.run_id.rsplit("_", maxsplit=1)[-1],
        "config_path": str(run.config_path),
        "resolved_config": config.as_dict(),
        "update": update,
        "environment_steps": environment_steps,
        "best_evaluation_score": best_evaluation_score,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "observation_normalizer": (
            None
            if observation_normalizer is None
            else observation_normalizer.state_dict()
        ),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
