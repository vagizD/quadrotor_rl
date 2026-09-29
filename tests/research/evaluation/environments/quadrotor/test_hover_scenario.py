from dataclasses import replace

import numpy as np

from environments import get_hover_scenarios
from research.pipeline.control.residual.models.experiments import load_experiment_config
from geometry.math.rotation import Quaternion


CONFIG_PATH = "configs/ppo_hover_baseline.toml"


def test_hover_scenarios_change_only_position() -> None:
    config = load_experiment_config(CONFIG_PATH)
    scenarios = get_hover_scenarios()

    assert [scenario.name for scenario in scenarios] == [
        "at_target",
        "near_target",
        "far_target",
    ]
    expected_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    for scenario, expected_offset in zip(scenarios, expected_offsets):
        state = scenario.initial_state(
            config.environment.target_position,
            config.environment.target_heading,
        )
        np.testing.assert_allclose(
            state.position,
            config.environment.target_position + expected_offset,
        )
        np.testing.assert_array_equal(state.velocity, np.zeros(3))
        np.testing.assert_array_equal(state.angular_velocity, np.zeros(3))
        np.testing.assert_allclose(state.quaternion.as_array(), Quaternion.identity().as_array())


def test_hover_scenario_matches_nonzero_target_heading() -> None:
    config = load_experiment_config(CONFIG_PATH)
    target_heading = np.pi / 3.0
    scenario = get_hover_scenarios()[0]

    state = scenario.initial_state(
        config.environment.target_position,
        target_heading,
    )

    assert np.isclose(state.quaternion.to_yaw(), target_heading)
    np.testing.assert_allclose(state.quaternion.to_rotation_matrix()[:, 2], [0.0, 0.0, 1.0])
