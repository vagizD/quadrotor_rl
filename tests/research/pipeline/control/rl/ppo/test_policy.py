import torch
import pytest

from research.pipeline.control.residual.models.experiments import load_experiment_config
from research.pipeline.control.rl.ppo import Actor, Critic


CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def make_actor(config=None) -> Actor:
    config = load_experiment_config(CONFIG_PATH) if config is None else config
    return Actor(
        config.dimensions.observation_dim,
        config.dimensions.action_dim,
        config.ppo.hidden_sizes,
        config.ppo.initial_log_std,
        config.ppo.min_log_std,
        config.ppo.max_log_std,
    )


def test_actor_returns_one_mean_per_action() -> None:
    config = load_experiment_config(CONFIG_PATH)
    actor = make_actor(config)

    means = actor(torch.zeros(3, config.dimensions.observation_dim))

    assert means.shape == (3, config.dimensions.action_dim)
    assert torch.isfinite(means).all()


def test_actor_uses_configured_policy_std_bounds() -> None:
    config = load_experiment_config(CONFIG_PATH)
    actor = make_actor(config)

    expected = torch.exp(
        torch.full(
            (config.dimensions.action_dim,),
            config.ppo.initial_log_std,
        )
    )

    torch.testing.assert_close(actor.policy_std, expected)
    assert torch.all(actor.policy_std >= torch.exp(torch.tensor(config.ppo.min_log_std)))
    assert torch.all(actor.policy_std <= torch.exp(torch.tensor(config.ppo.max_log_std)))


def test_bounded_gaussian_samples_valid_thrusts_and_scores_them() -> None:
    config = load_experiment_config(CONFIG_PATH)
    actor = make_actor(config)
    max_thrust = config.quadrotor.max_thrust
    distribution = actor.distribution(
        torch.zeros(8, config.dimensions.observation_dim),
        action_low=0.0,
        action_high=max_thrust,
    )

    actions = distribution.sample()
    log_probabilities = distribution.log_prob(actions)

    assert actions.shape == (8, config.dimensions.action_dim)
    assert torch.all(actions > 0.0)
    assert torch.all(actions < max_thrust)
    assert log_probabilities.shape == (8,)
    assert torch.isfinite(log_probabilities).all()


def test_bounded_gaussian_mean_is_inside_action_bounds() -> None:
    config = load_experiment_config(CONFIG_PATH)
    actor = make_actor(config)
    max_thrust = config.quadrotor.max_thrust
    mean_action = actor.distribution(
        torch.zeros(config.dimensions.observation_dim),
        action_low=0.0,
        action_high=max_thrust,
    ).mean

    assert mean_action.shape == (config.dimensions.action_dim,)
    assert torch.all(mean_action > 0.0)
    assert torch.all(mean_action < max_thrust)


def test_critic_returns_one_value_per_observation() -> None:
    config = load_experiment_config(CONFIG_PATH)
    critic = Critic(config.dimensions.observation_dim, config.ppo.hidden_sizes)

    values = critic(torch.zeros(5, config.dimensions.observation_dim))
    single_value = critic(torch.zeros(config.dimensions.observation_dim))

    assert values.shape == (5,)
    assert single_value.ndim == 0
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("dimension_offset", [-1, 1])
def test_actor_rejects_wrong_observation_dimension(dimension_offset: int) -> None:
    config = load_experiment_config(CONFIG_PATH)
    actor = make_actor(config)
    bad_dimension = config.dimensions.observation_dim + dimension_offset

    with pytest.raises(ValueError, match="last dimension"):
        actor(torch.zeros(bad_dimension))


def test_gated_dual_policy_outputs_nominal_hover_thrust_at_origin() -> None:
    actor = Actor(
        observation_dim=14,
        action_dim=4,
        hidden_sizes=(128, 128),
        initial_log_std=-3.0,
        min_log_std=-4.0,
        max_log_std=-1.0,
        gated_dual_policy=True,
    )
    # At w_hover = 1.0 (origin), the gated dual policy should output 2.4525 N on all 4 motors
    obs_at_origin = torch.zeros(14)
    obs_at_origin[-1] = 1.0  # w_hover = 1.0

    dist = actor.distribution(obs_at_origin, action_low=0.0, action_high=5.0)

    assert dist.mean.shape == (4,)
    torch.testing.assert_close(
        dist.mean,
        torch.full((4,), 2.4525),
        rtol=1e-4,
        atol=1e-4,
    )
