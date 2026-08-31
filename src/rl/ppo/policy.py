"""PPO actor, bounded action distribution, and critic."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("network dimensions must be positive")
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_sizes:
        if hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.Tanh()])
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def _validate_observation(observation: Tensor, observation_dim: int) -> Tensor:
    if not isinstance(observation, Tensor):
        raise TypeError("observation must be a torch.Tensor")
    if observation.ndim not in (1, 2):
        raise ValueError("observation must have shape (D,) or (B, D)")
    if observation.shape[-1] != observation_dim:
        raise ValueError(
            f"observation must have last dimension {observation_dim}, "
            f"got {observation.shape[-1]}"
        )
    if not observation.is_floating_point():
        raise TypeError("observation must use a floating-point dtype")
    return observation


class Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        initial_log_std: float,
        min_log_std: float,
        max_log_std: float,
    ) -> None:
        super().__init__()
        if min_log_std > max_log_std:
            raise ValueError("min_log_std cannot exceed max_log_std")
        if not min_log_std <= initial_log_std <= max_log_std:
            raise ValueError(
                "initial_log_std must be within [min_log_std, max_log_std]"
            )
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.mean_network = _build_mlp(
            observation_dim,
            action_dim,
            hidden_sizes,
        )
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(initial_log_std))
        )

    @property
    def effective_log_std(self) -> Tensor:
        """Return the bounded log standard deviations used for sampling."""
        return self.log_std.clamp(self.min_log_std, self.max_log_std)

    @property
    def policy_std(self) -> Tensor:
        """Return the learned pre-squash standard deviations."""
        return self.effective_log_std.exp()

    def forward(self, observation: Tensor) -> Tensor:
        observation = _validate_observation(observation, self.observation_dim)
        return self.mean_network(observation)

    def distribution(
        self,
        observation: Tensor,
        action_low: float | Tensor,
        action_high: float | Tensor,
    ) -> "SquashedGaussian":
        means = self(observation)
        return SquashedGaussian(
            means,
            self.effective_log_std,
            action_low,
            action_high,
        )


class SquashedGaussian:
    def __init__(
        self,
        mean: Tensor,
        log_std: Tensor,
        action_low: float | Tensor,
        action_high: float | Tensor,
    ) -> None:
        if mean.ndim < 1:
            raise ValueError("mean must have an action dimension")
        if log_std.ndim != 1 or log_std.shape[0] != mean.shape[-1]:
            raise ValueError("log_std must match the action dimension")
        self.mean_unbounded = mean
        self.log_std = log_std
        self.action_low = torch.as_tensor(
            action_low,
            dtype=mean.dtype,
            device=mean.device,
        )
        self.action_high = torch.as_tensor(
            action_high,
            dtype=mean.dtype,
            device=mean.device,
        )
        if self.action_low.ndim > 1 or self.action_high.ndim > 1:
            raise ValueError("action bounds must be scalars or one-dimensional")
        if self.action_low.numel() not in (1, mean.shape[-1]):
            raise ValueError("action_low must be scalar or match action dimension")
        if self.action_high.numel() not in (1, mean.shape[-1]):
            raise ValueError("action_high must be scalar or match action dimension")
        if torch.any(self.action_high <= self.action_low):
            raise ValueError("action_high must exceed action_low")
        self._scale = (self.action_high - self.action_low) / 2.0
        self._bias = (self.action_high + self.action_low) / 2.0
        self._normal = torch.distributions.Normal(mean, log_std.exp())

    @property
    def mean(self) -> Tensor:
        return self._transform(self.mean_unbounded)

    def sample(self) -> Tensor:
        return self._transform(self._normal.sample())

    def rsample(self) -> Tensor:
        return self._transform(self._normal.rsample())

    def log_prob(self, action: Tensor) -> Tensor:
        if not isinstance(action, Tensor):
            raise TypeError("action must be a torch.Tensor")
        normalized = (action - self._bias) / self._scale
        epsilon = torch.finfo(action.dtype).eps
        normalized = normalized.clamp(-1.0 + epsilon, 1.0 - epsilon)
        pre_squash = torch.atanh(normalized)
        log_abs_det = torch.log(self._scale) + 2.0 * (
            math.log(2.0)
            - pre_squash
            - F.softplus(-2.0 * pre_squash)
        )
        return (self._normal.log_prob(pre_squash) - log_abs_det).sum(dim=-1)

    def entropy(self) -> Tensor:
        # Tanh transforms do not have a simple closed-form entropy. A
        # reparameterized sample gives the optional PPO entropy estimate.
        action = self.rsample()
        return -self.log_prob(action)

    def _transform(self, pre_squash: Tensor) -> Tensor:
        return self._bias + self._scale * torch.tanh(pre_squash)


class Critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.value_network = _build_mlp(observation_dim, 1, hidden_sizes)

    def forward(self, observation: Tensor) -> Tensor:
        observation = _validate_observation(observation, self.observation_dim)
        return self.value_network(observation).squeeze(-1)
