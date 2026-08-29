import numpy as np
import pytest

from geometry.rotation import Quaternion
from robots.quadrotor.state import QuadrotorState


def test_state_components_have_expected_shapes() -> None:
    state = QuadrotorState(
        position=[1.0, 2.0, 3.0],
        velocity=[0.0, 0.0, 0.0],
        quaternion=Quaternion.identity(),
        angular_velocity=[0.0, 0.0, 0.0],
    )

    assert state.position.shape == (3,)
    assert state.velocity.shape == (3,)
    assert state.quaternion.as_array().shape == (4,)
    assert state.angular_velocity.shape == (3,)
    assert isinstance(state.quaternion, Quaternion)
    assert all(
        isinstance(getattr(state, name), np.ndarray)
        for name in ("position", "velocity", "angular_velocity")
    )


@pytest.mark.parametrize("field", ["position", "velocity", "angular_velocity"])
def test_three_component_fields_reject_wrong_shape(field: str) -> None:
    values = dict(
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        quaternion=Quaternion.identity(),
        angular_velocity=[0.0, 0.0, 0.0],
    )
    values[field] = [0.0, 0.0]

    with pytest.raises(ValueError, match=field):
        QuadrotorState(**values)


def test_quaternion_field_requires_quaternion_class() -> None:
    with pytest.raises(TypeError, match="Quaternion"):
        QuadrotorState(
            position=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            quaternion=np.ones(4),
            angular_velocity=[0.0, 0.0, 0.0],
        )
