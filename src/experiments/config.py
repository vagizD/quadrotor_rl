"""Loading and snapshotting the experiment configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import numpy as np

from common.dimensions import QDIMS
from common.types import FloatVector

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - used on Python 3.10
    import tomli as tomllib

from robots.quadrotor import QuadrotorParams

IntegratorName = Literal["euler", "rk4"]
InitializerName = Literal["random"]
CONFIG_SECTIONS = {
    "experiment",
    "quadrotor",
    "simulation",
    "dimensions",
    "environment",
    "reward",
    "algorithm",
    "ppo",
    "tracking",
    "evaluation",
    "training",
}


def _require_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_limit(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_weight(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _nonnegative_limit(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _finite_angle(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _unit_interval(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class SimulationConfig:
    dt: float
    integrator: IntegratorName

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt", _positive_limit(self.dt, "simulation dt"))
        if (
            not isinstance(self.integrator, str)
            or self.integrator not in ("euler", "rk4")
        ):
            raise ValueError("simulation integrator must be 'euler' or 'rk4'")


@dataclass(frozen=True)
class DimensionsConfig:
    observation_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_dim", self.observation_dim),
            ("action_dim", self.action_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class RandomInitializerConfig:
    name: InitializerName
    min_position_distance: float
    max_position_distance: float
    min_velocity: float
    max_velocity: float
    min_tilt_angle: float
    max_tilt_angle: float
    min_angular_velocity: float
    max_angular_velocity: float

    def __post_init__(self) -> None:
        if self.name != "random":
            raise ValueError("initializer name must be 'random'")
        ranges = (
            ("min_position_distance", self.min_position_distance),
            ("max_position_distance", self.max_position_distance),
            ("min_velocity", self.min_velocity),
            ("max_velocity", self.max_velocity),
            ("min_tilt_angle", self.min_tilt_angle),
            ("max_tilt_angle", self.max_tilt_angle),
            ("min_angular_velocity", self.min_angular_velocity),
            ("max_angular_velocity", self.max_angular_velocity),
        )
        for name, value in ranges:
            object.__setattr__(self, name, _nonnegative_limit(value, name))
        for minimum_name, maximum_name in (
            ("min_position_distance", "max_position_distance"),
            ("min_velocity", "max_velocity"),
            ("min_tilt_angle", "max_tilt_angle"),
            ("min_angular_velocity", "max_angular_velocity"),
        ):
            if getattr(self, minimum_name) > getattr(self, maximum_name):
                raise ValueError(f"{minimum_name} cannot exceed {maximum_name}")
        if self.max_tilt_angle > np.pi:
            raise ValueError("max_tilt_angle must not exceed pi")


@dataclass(frozen=True)
class HoverSuccessConfig:
    position_error: float
    speed: float
    angular_rate: float
    tilt_angle: float
    heading_error: float

    def __post_init__(self) -> None:
        for name in (
            "position_error",
            "speed",
            "angular_rate",
            "tilt_angle",
            "heading_error",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_limit(getattr(self, name), f"success {name}"),
            )
        if self.tilt_angle > np.pi:
            raise ValueError("success tilt_angle must not exceed pi")
        if self.heading_error > np.pi:
            raise ValueError("success heading_error must not exceed pi")


@dataclass(frozen=True)
class HoverEnvironmentConfig:
    target_position: FloatVector
    target_heading: float
    episode_horizon: float
    max_position_error: float
    max_tilt_angle: float
    success: HoverSuccessConfig
    initializer: RandomInitializerConfig

    def __post_init__(self) -> None:
        target = np.array(self.target_position, dtype=np.float64, copy=True)
        if target.shape != (3,):
            raise ValueError(
                f"target_position must have shape (3,), got {target.shape}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_position must contain only finite values")
        object.__setattr__(self, "target_position", target)
        object.__setattr__(
            self,
            "target_heading",
            _finite_angle(self.target_heading, "target_heading"),
        )
        object.__setattr__(
            self,
            "episode_horizon",
            _positive_limit(self.episode_horizon, "episode_horizon"),
        )
        object.__setattr__(
            self,
            "max_position_error",
            _positive_limit(self.max_position_error, "max_position_error"),
        )
        if (
            isinstance(self.max_tilt_angle, bool)
            or not isinstance(self.max_tilt_angle, Real)
            or not np.isfinite(self.max_tilt_angle)
            or not 0.0 < self.max_tilt_angle <= np.pi
        ):
            raise ValueError("max_tilt_angle must be in (0, pi]")
        object.__setattr__(self, "max_tilt_angle", float(self.max_tilt_angle))
        if not isinstance(self.success, HoverSuccessConfig):
            raise TypeError("success must be HoverSuccessConfig")
        if not isinstance(self.initializer, RandomInitializerConfig):
            raise TypeError("initializer must be RandomInitializerConfig")


@dataclass(frozen=True)
class RewardConfig:
    alive_reward: float
    position_error_weight: float
    linear_velocity_weight: float
    tilt_weight: float
    heading_weight: float
    angular_velocity_weight: float
    thrust_deviation_weight: float
    terminal_penalty: float
    truncation_distance_scale: float
    cost_normalization_scale: float
    normalized_cost_limit: float

    def __post_init__(self) -> None:
        for name in (
            "alive_reward",
            "position_error_weight",
            "linear_velocity_weight",
            "tilt_weight",
            "heading_weight",
            "angular_velocity_weight",
            "thrust_deviation_weight",
            "terminal_penalty",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_weight(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "cost_normalization_scale",
            _positive_limit(
                self.cost_normalization_scale,
                "cost_normalization_scale",
            ),
        )
        object.__setattr__(
            self,
            "truncation_distance_scale",
            _positive_limit(
                self.truncation_distance_scale,
                "truncation_distance_scale",
            ),
        )
        object.__setattr__(
            self,
            "normalized_cost_limit",
            _unit_interval(
                self.normalized_cost_limit,
                "normalized_cost_limit",
            ),
        )
        if self.normalized_cost_limit <= 0.0:
            raise ValueError("normalized_cost_limit must be positive")
        if self.normalized_cost_limit >= self.alive_reward:
            raise ValueError(
                "normalized_cost_limit must be less than alive_reward"
            )


@dataclass(frozen=True)
class TrackingConfig:
    console_log_interval: int
    rerun_step_stride: int
    record_evaluation_rerun: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "console_log_interval",
            _positive_integer(
                self.console_log_interval,
                "tracking console_log_interval",
            ),
        )
        object.__setattr__(
            self,
            "rerun_step_stride",
            _positive_integer(self.rerun_step_stride, "tracking rerun_step_stride"),
        )
        if not isinstance(self.record_evaluation_rerun, bool):
            raise ValueError("tracking record_evaluation_rerun must be a boolean")


@dataclass(frozen=True)
class EvaluationConfig:
    interval_updates: int
    episodes: int
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interval_updates",
            _positive_integer(self.interval_updates, "evaluation interval_updates"),
        )
        object.__setattr__(
            self,
            "episodes",
            _positive_integer(self.episodes, "evaluation episodes"),
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError("evaluation seed must be a non-negative integer")
        object.__setattr__(self, "seed", int(self.seed))
@dataclass(frozen=True)
class TrainingConfig:
    updates: int
    rolling_episode_window: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "updates",
            _positive_integer(self.updates, "training updates"),
        )
        object.__setattr__(
            self,
            "rolling_episode_window",
            _positive_integer(
                self.rolling_episode_window,
                "training rolling_episode_window",
            ),
        )


@dataclass(frozen=True)
class PPOConfig:
    hidden_sizes: tuple[int, ...]
    initial_log_std: float
    min_log_std: float
    max_log_std: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    learning_rate: float
    value_loss_coef: float
    entropy_coef: float
    rollout_steps: int
    steps_per_rollout: int
    normalize_observations: bool
    observation_clip: float
    normalize_advantages: bool
    normalization_epsilon: float
    reward_scale: float
    max_grad_norm: float

    def __post_init__(self) -> None:
        hidden_sizes = tuple(self.hidden_sizes)
        if not hidden_sizes or any(
            isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
            for size in hidden_sizes
        ):
            raise ValueError("ppo hidden_sizes must contain positive integers")
        object.__setattr__(
            self,
            "hidden_sizes",
            tuple(int(size) for size in hidden_sizes),
        )
        if (
            isinstance(self.initial_log_std, bool)
            or not isinstance(self.initial_log_std, Real)
            or not np.isfinite(self.initial_log_std)
        ):
            raise ValueError("ppo initial_log_std must be finite")
        object.__setattr__(self, "initial_log_std", float(self.initial_log_std))
        for name in ("min_log_std", "max_log_std"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"ppo {name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.min_log_std > self.max_log_std:
            raise ValueError("ppo min_log_std cannot exceed max_log_std")
        if not self.min_log_std <= self.initial_log_std <= self.max_log_std:
            raise ValueError(
                "ppo initial_log_std must be within [min_log_std, max_log_std]"
            )
        object.__setattr__(self, "gamma", _unit_interval(self.gamma, "ppo gamma"))
        object.__setattr__(
            self,
            "gae_lambda",
            _unit_interval(self.gae_lambda, "ppo gae_lambda"),
        )
        object.__setattr__(
            self,
            "clip_epsilon",
            _positive_limit(self.clip_epsilon, "ppo clip_epsilon"),
        )
        object.__setattr__(
            self,
            "learning_rate",
            _positive_limit(self.learning_rate, "ppo learning_rate"),
        )
        object.__setattr__(
            self,
            "value_loss_coef",
            _nonnegative_weight(self.value_loss_coef, "ppo value_loss_coef"),
        )
        object.__setattr__(
            self,
            "entropy_coef",
            _nonnegative_weight(self.entropy_coef, "ppo entropy_coef"),
        )
        object.__setattr__(
            self,
            "rollout_steps",
            _positive_integer(self.rollout_steps, "ppo rollout_steps"),
        )
        object.__setattr__(
            self,
            "steps_per_rollout",
            _positive_integer(
                self.steps_per_rollout,
                "ppo steps_per_rollout",
            ),
        )
        for name in ("normalize_observations", "normalize_advantages"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"ppo {name} must be a boolean")
        object.__setattr__(
            self,
            "observation_clip",
            _positive_limit(self.observation_clip, "ppo observation_clip"),
        )
        object.__setattr__(
            self,
            "normalization_epsilon",
            _positive_limit(
                self.normalization_epsilon,
                "ppo normalization_epsilon",
            ),
        )
        object.__setattr__(
            self,
            "reward_scale",
            _positive_limit(self.reward_scale, "ppo reward_scale"),
        )
        object.__setattr__(
            self,
            "max_grad_norm",
            _positive_limit(self.max_grad_norm, "ppo max_grad_norm"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    algorithm: str
    quadrotor: QuadrotorParams
    simulation: SimulationConfig
    dimensions: DimensionsConfig
    environment: HoverEnvironmentConfig
    reward: RewardConfig
    ppo: PPOConfig
    tracking: TrackingConfig
    evaluation: EvaluationConfig
    training: TrainingConfig

    def __post_init__(self) -> None:
        _require_name(self.name, "experiment name")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError("experiment seed must be a non-negative integer")
        _require_name(self.algorithm, "algorithm name")
        if not isinstance(self.quadrotor, QuadrotorParams):
            raise TypeError("quadrotor must be QuadrotorParams")
        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("simulation must be SimulationConfig")
        if not isinstance(self.dimensions, DimensionsConfig):
            raise TypeError("dimensions must be DimensionsConfig")
        if self.dimensions.observation_dim != QDIMS.observation_dim:
            raise ValueError(
                "dimensions observation_dim must match the current quadrotor state "
                f"layout ({QDIMS.observation_dim})"
            )
        if self.dimensions.action_dim != QDIMS.action_dim:
            raise ValueError(
                "dimensions action_dim must match the current quadrotor actuator "
                f"layout ({QDIMS.action_dim})"
            )
        if self.dimensions.action_dim != self.quadrotor.motor_spin_directions.size:
            raise ValueError(
                "dimensions action_dim must match quadrotor motor_spin_directions"
            )
        if not isinstance(self.environment, HoverEnvironmentConfig):
            raise TypeError("environment must be HoverEnvironmentConfig")
        if not isinstance(self.reward, RewardConfig):
            raise TypeError("reward must be RewardConfig")
        if not isinstance(self.ppo, PPOConfig):
            raise TypeError("ppo must be PPOConfig")
        if not isinstance(self.tracking, TrackingConfig):
            raise TypeError("tracking must be TrackingConfig")
        if not isinstance(self.evaluation, EvaluationConfig):
            raise TypeError("evaluation must be EvaluationConfig")
        if not isinstance(self.training, TrainingConfig):
            raise TypeError("training must be TrainingConfig")
        object.__setattr__(self, "seed", int(self.seed))

    def as_dict(self) -> dict[str, dict[str, object]]:
        params = self.quadrotor
        return {
            "experiment": {"name": self.name, "seed": self.seed},
            "quadrotor": {
                "mass": params.mass,
                "arm_length": params.arm_length,
                "inertia": params.inertia.tolist(),
                "yaw_torque_coefficient": params.yaw_torque_coefficient,
                "max_thrust": params.max_thrust,
                "gravity": params.gravity,
                "motor_spin_directions": params.motor_spin_directions.tolist(),
            },
            "simulation": {
                "dt": self.simulation.dt,
                "integrator": self.simulation.integrator,
            },
            "dimensions": {
                "observation_dim": self.dimensions.observation_dim,
                "action_dim": self.dimensions.action_dim,
            },
            "environment": {
                "target_position": self.environment.target_position.tolist(),
                "target_heading": self.environment.target_heading,
                "episode_horizon": self.environment.episode_horizon,
                "max_position_error": self.environment.max_position_error,
                "max_tilt_angle": self.environment.max_tilt_angle,
                "success": {
                    "position_error": self.environment.success.position_error,
                    "speed": self.environment.success.speed,
                    "angular_rate": self.environment.success.angular_rate,
                    "tilt_angle": self.environment.success.tilt_angle,
                    "heading_error": self.environment.success.heading_error,
                },
                "initializer": {
                    "name": self.environment.initializer.name,
                    "min_position_distance": (
                        self.environment.initializer.min_position_distance
                    ),
                    "max_position_distance": (
                        self.environment.initializer.max_position_distance
                    ),
                    "min_velocity": self.environment.initializer.min_velocity,
                    "max_velocity": self.environment.initializer.max_velocity,
                    "min_tilt_angle": self.environment.initializer.min_tilt_angle,
                    "max_tilt_angle": self.environment.initializer.max_tilt_angle,
                    "min_angular_velocity": (
                        self.environment.initializer.min_angular_velocity
                    ),
                    "max_angular_velocity": (
                        self.environment.initializer.max_angular_velocity
                    ),
                },
            },
            "reward": {
                "alive_reward": self.reward.alive_reward,
                "position_error_weight": self.reward.position_error_weight,
                "linear_velocity_weight": self.reward.linear_velocity_weight,
                "tilt_weight": self.reward.tilt_weight,
                "heading_weight": self.reward.heading_weight,
                "angular_velocity_weight": self.reward.angular_velocity_weight,
                "thrust_deviation_weight": self.reward.thrust_deviation_weight,
                "terminal_penalty": self.reward.terminal_penalty,
                "truncation_distance_scale": self.reward.truncation_distance_scale,
                "cost_normalization_scale": self.reward.cost_normalization_scale,
                "normalized_cost_limit": self.reward.normalized_cost_limit,
            },
            "algorithm": {"name": self.algorithm},
            "ppo": {
                "hidden_sizes": list(self.ppo.hidden_sizes),
                "initial_log_std": self.ppo.initial_log_std,
                "min_log_std": self.ppo.min_log_std,
                "max_log_std": self.ppo.max_log_std,
                "gamma": self.ppo.gamma,
                "gae_lambda": self.ppo.gae_lambda,
                "clip_epsilon": self.ppo.clip_epsilon,
                "learning_rate": self.ppo.learning_rate,
                "value_loss_coef": self.ppo.value_loss_coef,
                "entropy_coef": self.ppo.entropy_coef,
                "rollout_steps": self.ppo.rollout_steps,
                "steps_per_rollout": self.ppo.steps_per_rollout,
                "normalize_observations": self.ppo.normalize_observations,
                "observation_clip": self.ppo.observation_clip,
                "normalize_advantages": self.ppo.normalize_advantages,
                "normalization_epsilon": self.ppo.normalization_epsilon,
                "reward_scale": self.ppo.reward_scale,
                "max_grad_norm": self.ppo.max_grad_norm,
            },
            "tracking": {
                "console_log_interval": self.tracking.console_log_interval,
                "rerun_step_stride": self.tracking.rerun_step_stride,
                "record_evaluation_rerun": self.tracking.record_evaluation_rerun,
            },
            "evaluation": {
                "interval_updates": self.evaluation.interval_updates,
                "episodes": self.evaluation.episodes,
                "seed": self.evaluation.seed,
            },
            "training": {
                "updates": self.training.updates,
                "rolling_episode_window": self.training.rolling_episode_window,
            },
        }


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _check_keys(section: Mapping[str, object], name: str, required: set[str]) -> None:
    missing = required - set(section)
    unknown = set(section) - required
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"invalid {name} section: {', '.join(details)}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate the complete experiment settings."""
    with Path(path).open("rb") as file:
        raw = tomllib.load(file)
    if set(raw) != CONFIG_SECTIONS:
        raise ValueError(
            f"config must contain exactly {sorted(CONFIG_SECTIONS)} sections"
        )

    experiment_data = _require_mapping(raw["experiment"], "experiment")
    _check_keys(experiment_data, "experiment", {"name", "seed"})
    algorithm_data = _require_mapping(raw["algorithm"], "algorithm")
    _check_keys(algorithm_data, "algorithm", {"name"})

    quadrotor_data = _require_mapping(raw["quadrotor"], "quadrotor")
    _check_keys(
        quadrotor_data,
        "quadrotor",
        {
            "mass",
            "arm_length",
            "inertia",
            "yaw_torque_coefficient",
            "max_thrust",
            "gravity",
            "motor_spin_directions",
        },
    )
    try:
        params = QuadrotorParams(
            mass=quadrotor_data["mass"],
            arm_length=quadrotor_data["arm_length"],
            inertia=np.asarray(quadrotor_data["inertia"], dtype=np.float64),
            yaw_torque_coefficient=quadrotor_data["yaw_torque_coefficient"],
            max_thrust=quadrotor_data["max_thrust"],
            gravity=quadrotor_data["gravity"],
            motor_spin_directions=np.asarray(
                quadrotor_data["motor_spin_directions"],
                dtype=np.float64,
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid quadrotor configuration: {error}") from error

    simulation_data = _require_mapping(raw["simulation"], "simulation")
    _check_keys(simulation_data, "simulation", {"dt", "integrator"})
    simulation = SimulationConfig(
        dt=simulation_data["dt"],
        integrator=simulation_data["integrator"],
    )

    dimensions_data = _require_mapping(raw["dimensions"], "dimensions")
    _check_keys(dimensions_data, "dimensions", {"observation_dim", "action_dim"})
    dimensions = DimensionsConfig(
        observation_dim=dimensions_data["observation_dim"],
        action_dim=dimensions_data["action_dim"],
    )

    environment_data = _require_mapping(raw["environment"], "environment")
    _check_keys(
        environment_data,
        "environment",
        {
            "target_position",
            "target_heading",
            "episode_horizon",
            "max_position_error",
            "max_tilt_angle",
            "success",
            "initializer",
        },
    )
    success_data = _require_mapping(
        environment_data["success"],
        "environment.success",
    )
    _check_keys(
        success_data,
        "environment.success",
        {"position_error", "speed", "angular_rate", "tilt_angle", "heading_error"},
    )
    initializer_data = _require_mapping(
        environment_data["initializer"],
        "environment.initializer",
    )
    _check_keys(
        initializer_data,
        "environment.initializer",
        {
            "name",
            "min_position_distance",
            "max_position_distance",
            "min_velocity",
            "max_velocity",
            "min_tilt_angle",
            "max_tilt_angle",
            "min_angular_velocity",
            "max_angular_velocity",
        },
    )
    environment = HoverEnvironmentConfig(
        target_position=np.asarray(
            environment_data["target_position"],
            dtype=np.float64,
        ),
        target_heading=environment_data["target_heading"],
        episode_horizon=environment_data["episode_horizon"],
        max_position_error=environment_data["max_position_error"],
        max_tilt_angle=environment_data["max_tilt_angle"],
        success=HoverSuccessConfig(**success_data),
        initializer=RandomInitializerConfig(**initializer_data),
    )

    reward_data = _require_mapping(raw["reward"], "reward")
    _check_keys(
        reward_data,
        "reward",
        {
            "alive_reward",
            "position_error_weight",
            "linear_velocity_weight",
            "tilt_weight",
            "heading_weight",
            "angular_velocity_weight",
            "thrust_deviation_weight",
            "terminal_penalty",
            "truncation_distance_scale",
            "cost_normalization_scale",
            "normalized_cost_limit",
        },
    )
    reward = RewardConfig(**reward_data)
    tracking_data = _require_mapping(raw["tracking"], "tracking")
    _check_keys(
        tracking_data,
        "tracking",
        {
            "console_log_interval",
            "rerun_step_stride",
            "record_evaluation_rerun",
        },
    )
    tracking = TrackingConfig(**tracking_data)
    evaluation_data = _require_mapping(raw["evaluation"], "evaluation")
    _check_keys(
        evaluation_data,
        "evaluation",
        {
            "interval_updates",
            "episodes",
            "seed",
        },
    )
    evaluation = EvaluationConfig(**evaluation_data)
    training_data = _require_mapping(raw["training"], "training")
    _check_keys(
        training_data,
        "training",
        {"updates", "rolling_episode_window"},
    )
    training = TrainingConfig(**training_data)
    ppo_data = _require_mapping(raw["ppo"], "ppo")
    _check_keys(
        ppo_data,
        "ppo",
        {
            "hidden_sizes",
            "initial_log_std",
            "min_log_std",
            "max_log_std",
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "learning_rate",
            "value_loss_coef",
            "entropy_coef",
            "rollout_steps",
            "steps_per_rollout",
            "normalize_observations",
            "observation_clip",
            "normalize_advantages",
            "normalization_epsilon",
            "reward_scale",
            "max_grad_norm",
        },
    )
    ppo = PPOConfig(
        hidden_sizes=tuple(ppo_data["hidden_sizes"]),
        initial_log_std=ppo_data["initial_log_std"],
        min_log_std=ppo_data["min_log_std"],
        max_log_std=ppo_data["max_log_std"],
        gamma=ppo_data["gamma"],
        gae_lambda=ppo_data["gae_lambda"],
        clip_epsilon=ppo_data["clip_epsilon"],
        learning_rate=ppo_data["learning_rate"],
        value_loss_coef=ppo_data["value_loss_coef"],
        entropy_coef=ppo_data["entropy_coef"],
        rollout_steps=ppo_data["rollout_steps"],
        steps_per_rollout=ppo_data["steps_per_rollout"],
        normalize_observations=ppo_data["normalize_observations"],
        observation_clip=ppo_data["observation_clip"],
        normalize_advantages=ppo_data["normalize_advantages"],
        normalization_epsilon=ppo_data["normalization_epsilon"],
        reward_scale=ppo_data["reward_scale"],
        max_grad_norm=ppo_data["max_grad_norm"],
    )
    return ExperimentConfig(
        name=experiment_data["name"],
        seed=experiment_data["seed"],
        algorithm=algorithm_data["name"],
        quadrotor=params,
        simulation=simulation,
        dimensions=dimensions,
        environment=environment,
        reward=reward,
        ppo=ppo,
        tracking=tracking,
        evaluation=evaluation,
        training=training,
    )


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def _toml_lines(section: str, values: Mapping[str, object]) -> list[str]:
    lines = [f"[{section}]"]
    nested: list[tuple[str, Mapping[str, object]]] = []
    for key, value in values.items():
        if isinstance(value, Mapping):
            nested.append((key, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    for key, value in nested:
        lines.extend(_toml_lines(f"{section}.{key}", value))
    return lines


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write a canonical TOML snapshot of a validated config."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for section, values in config.as_dict().items():
        lines.extend(_toml_lines(section, values))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
