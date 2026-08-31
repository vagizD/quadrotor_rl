import numpy as np
import pytest

from common.dimensions import QDIMS
from geometry.rotation import Quaternion
from robots.quadrotor.state import QuadrotorState


def test_state_components_have_expected_shapes() -> None:
    state = QuadrotorState(
        position=[1.0, 2.0, 3.0],
        velocity=[0.0, 0.0, 0.0],
        quaternion=Quaternion.identity(),
        angular_velocity=[0.0, 0.0, 0.0],
    )

    assert state.position.shape == (QDIMS.position_dim,)
    assert state.velocity.shape == (QDIMS.velocity_dim,)
    assert state.quaternion.as_array().shape == (QDIMS.quaternion_dim,)
    assert state.angular_velocity.shape == (QDIMS.angular_velocity_dim,)
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
            quaternion=np.ones(QDIMS.quaternion_dim),
            angular_velocity=[0.0, 0.0, 0.0],
        )
