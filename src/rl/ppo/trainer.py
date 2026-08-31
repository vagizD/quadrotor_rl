"""PPO objective terms and one full-batch optimizer update."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from experiments import PPOConfig

from .policy import Actor, Critic
from .rollout import PPOBatch, PPORollout


def compute_probability_ratio(
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
) -> Tensor:
    """Compute PPO probability ratios from two log-probability vectors."""
    if not isinstance(new_log_probabilities, Tensor):
        raise TypeError("new_log_probabilities must be a torch.Tensor")
    if not isinstance(old_log_probabilities, Tensor):
        raise TypeError("old_log_probabilities must be a torch.Tensor")
    if new_log_probabilities.shape != old_log_probabilities.shape:
        raise ValueError("new and old log-probabilities must have the same shape")
    return torch.exp(new_log_probabilities - old_log_probabilities)


def compute_clipped_surrogate(
    ratios: Tensor,
    advantages: Tensor,
    clip_epsilon: float,
) -> Tensor:
    """Compute the per-transition clipped PPO surrogate terms."""
    if not isinstance(ratios, Tensor) or not isinstance(advantages, Tensor):
        raise TypeError("ratios and advantages must be torch.Tensor objects")
    if ratios.shape != advantages.shape:
        raise ValueError("ratios and advantages must have the same shape")
    if (
        isinstance(clip_epsilon, bool)
        or not isinstance(clip_epsilon, Real)
        or not np.isfinite(clip_epsilon)
        or clip_epsilon <= 0.0
    ):
        raise ValueError("clip_epsilon must be finite and positive")
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return torch.minimum(ratios * advantages, clipped_ratios * advantages)


@dataclass(frozen=True)
class PPOPolicyTerms:
    new_log_probabilities: Tensor
    ratios: Tensor
    clipped_ratios: Tensor
    policy_loss: Tensor
    entropy: Tensor
    clip_fraction: Tensor


@dataclass(frozen=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    mean_ratio: float
    clip_fraction: float
    approximate_kl: float
    explained_variance: float
    actor_grad_norm: float
    critic_grad_norm: float


class PPOTrainer:
    """Perform one full-batch PPO update for an actor and critic."""

    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        config: PPOConfig,
        action_low: float | Tensor,
        action_high: float | Tensor,
    ) -> None:
        if not isinstance(actor, Actor):
            raise TypeError("actor must be Actor")
        if not isinstance(critic, Critic):
            raise TypeError("critic must be Critic")
        if not isinstance(config, PPOConfig):
            raise TypeError("config must be PPOConfig")
        actor_parameter = next(actor.parameters())
        critic_parameter = next(critic.parameters())
        if (
            actor_parameter.dtype != critic_parameter.dtype
            or actor_parameter.device != critic_parameter.device
        ):
            raise ValueError("actor and critic must use the same dtype and device")
        self.actor = actor
        self.critic = critic
        self.config = config
        self.action_low = action_low
        self.action_high = action_high
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(),
            lr=config.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=config.learning_rate,
        )

    def compute_policy_terms(self, rollout: PPORollout) -> PPOPolicyTerms:
        """Evaluate the current policy against one PPO rollout."""
        self._prepare_targets(rollout)
        return self._compute_policy_terms(rollout.to_batch())

    def _compute_policy_terms(self, batch: PPOBatch) -> PPOPolicyTerms:
        observations, actions, old_log_probabilities, advantages, _ = (
            self._batch_tensors(batch)
        )
        distribution = self.actor.distribution(
            observations,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        new_log_probabilities = distribution.log_prob(actions)
        ratios = compute_probability_ratio(
            new_log_probabilities,
            old_log_probabilities,
        )
        clipped_ratios = ratios.clamp(
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )
        surrogate = compute_clipped_surrogate(
            ratios,
            advantages,
            self.config.clip_epsilon,
        )
        entropy = distribution.entropy().mean()
        clip_fraction = (ratios != clipped_ratios).to(torch.float32).mean()
        return PPOPolicyTerms(
            new_log_probabilities=new_log_probabilities,
            ratios=ratios,
            clipped_ratios=clipped_ratios,
            policy_loss=-surrogate.mean(),
            entropy=entropy,
            clip_fraction=clip_fraction,
        )

    def update(self, rollout: PPORollout) -> PPOUpdateMetrics:
        """Update actor and critic repeatedly from one complete rollout."""
        self._prepare_targets(rollout)
        batch = rollout.to_batch()
        observations, _, old_log_probabilities, _, returns = self._batch_tensors(batch)
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        total_loss_sum = 0.0
        mean_ratio_sum = 0.0
        clip_fraction_sum = 0.0
        approximate_kl_sum = 0.0
        explained_variance_sum = 0.0
        actor_grad_norm_sum = 0.0
        critic_grad_norm_sum = 0.0

        for _ in range(self.config.steps_per_rollout):
            policy_terms = self._compute_policy_terms(batch)
            values = self.critic(observations)
            value_loss = F.mse_loss(values, returns)
            return_variance = torch.var(returns, unbiased=False)
            explained_variance = torch.where(
                return_variance > torch.finfo(returns.dtype).eps,
                1.0
                - torch.var(returns - values.detach(), unbiased=False)
                / return_variance,
                torch.zeros_like(return_variance),
            )
            actor_loss = policy_terms.policy_loss - (
                self.config.entropy_coef * policy_terms.entropy
            )
            critic_objective = self.config.value_loss_coef * value_loss
            total_loss = actor_loss + critic_objective

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_objective.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.critic_optimizer.step()

            policy_loss_sum += float(policy_terms.policy_loss.detach().item())
            value_loss_sum += float(value_loss.detach().item())
            entropy_sum += float(policy_terms.entropy.detach().item())
            total_loss_sum += float(total_loss.detach().item())
            mean_ratio_sum += float(policy_terms.ratios.detach().mean().item())
            clip_fraction_sum += float(policy_terms.clip_fraction.detach().item())
            approximate_kl_sum += float(
                (old_log_probabilities - policy_terms.new_log_probabilities)
                .detach()
                .mean()
                .item()
            )
            explained_variance_sum += float(explained_variance.detach().item())
            actor_grad_norm_sum += float(actor_grad_norm.detach().item())
            critic_grad_norm_sum += float(critic_grad_norm.detach().item())

        steps = float(self.config.steps_per_rollout)

        return PPOUpdateMetrics(
            policy_loss=policy_loss_sum / steps,
            value_loss=value_loss_sum / steps,
            entropy=entropy_sum / steps,
            total_loss=total_loss_sum / steps,
            mean_ratio=mean_ratio_sum / steps,
            clip_fraction=clip_fraction_sum / steps,
            approximate_kl=approximate_kl_sum / steps,
            explained_variance=explained_variance_sum / steps,
            actor_grad_norm=actor_grad_norm_sum / steps,
            critic_grad_norm=critic_grad_norm_sum / steps,
        )

    def _prepare_targets(self, rollout: PPORollout) -> None:
        if not isinstance(rollout, PPORollout):
            raise TypeError("rollout must be PPORollout")
        if any(
            trajectory.returns is None or trajectory.advantages is None
            for trajectory in rollout.trajectories
        ):
            rollout.compute_targets(
                self.config.gamma,
                self.config.gae_lambda,
                self.config.reward_scale,
            )

    def _batch_tensors(
        self,
        batch: PPOBatch,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        parameter = next(self.actor.parameters())
        tensor = lambda value: torch.as_tensor(
            value,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        observations = tensor(batch.observations)
        actions = tensor(batch.actions)
        log_probabilities = tensor(batch.log_probabilities)
        advantages = tensor(batch.advantages)
        returns = tensor(batch.returns)
        if self.config.normalize_advantages:
            # Use one scale for the complete rollout batch, not per trajectory.
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + self.config.normalization_epsilon
            )
        return observations, actions, log_probabilities, advantages, returns
