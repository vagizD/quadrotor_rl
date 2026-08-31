"""Dimensions of the current quadrotor model."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class QuadrotorDimensions:
    position_dim: int = 3
    velocity_dim: int = 3
    quaternion_dim: int = 4
    angular_velocity_dim: int = 3
    motor_count: int = 4

    def __post_init__(self) -> None:
        for name in (
            "position_dim",
            "velocity_dim",
            "quaternion_dim",
            "angular_velocity_dim",
            "motor_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))

    @property
    def state_dim(self) -> int:
        return (
            self.position_dim
            + self.velocity_dim
            + self.quaternion_dim
            + self.angular_velocity_dim
        )

    @property
    def observation_dim(self) -> int:
        return self.state_dim

    @property
    def action_dim(self) -> int:
        return self.motor_count

    @property
    def position_slice(self) -> slice:
        return slice(0, self.position_dim)

    @property
    def velocity_slice(self) -> slice:
        start = self.position_slice.stop
        return slice(start, start + self.velocity_dim)

    @property
    def quaternion_slice(self) -> slice:
        start = self.velocity_slice.stop
        return slice(start, start + self.quaternion_dim)

    @property
    def angular_velocity_slice(self) -> slice:
        start = self.quaternion_slice.stop
        return slice(start, start + self.angular_velocity_dim)


QDIMS = QuadrotorDimensions()
