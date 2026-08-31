import numpy as np

from environments.quadrotor import RandomInitializer
from experiments import RandomInitializerConfig


def make_config() -> RandomInitializerConfig:
    return RandomInitializerConfig(
        name="random",
        min_position_distance=1.0,
        max_position_distance=2.0,
        min_velocity=3.0,
        max_velocity=4.0,
        min_tilt_angle=0.2,
        max_tilt_angle=0.4,
        min_angular_velocity=5.0,
        max_angular_velocity=6.0,
    )


def test_random_initializer_respects_configured_ranges() -> None:
    state = RandomInitializer(
        make_config(),
        np.array([10.0, 20.0, 30.0]),
        42,
    ).initialize()

    assert 1.0 <= np.linalg.norm(state.position - [10.0, 20.0, 30.0]) <= 2.0
    assert 3.0 <= np.linalg.norm(state.velocity) <= 4.0
    assert 5.0 <= np.linalg.norm(state.angular_velocity) <= 6.0
    body_up_world = state.quaternion.to_rotation_matrix()[:, 2]
    tilt_angle = np.arccos(np.clip(body_up_world[2], -1.0, 1.0))
    assert 0.2 <= tilt_angle <= 0.4


def test_random_initializer_zero_ranges_return_nominal_state() -> None:
    config = RandomInitializerConfig(
        name="random",
        min_position_distance=0.0,
        max_position_distance=0.0,
        min_velocity=0.0,
        max_velocity=0.0,
        min_tilt_angle=0.0,
        max_tilt_angle=0.0,
        min_angular_velocity=0.0,
        max_angular_velocity=0.0,
    )

    state = RandomInitializer(config, np.array([1.0, 2.0, 3.0]), 42).initialize()

    np.testing.assert_allclose(state.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state.velocity, np.zeros(3))
    np.testing.assert_allclose(state.quaternion.as_array(), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.angular_velocity, np.zeros(3))
