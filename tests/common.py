import numpy as np

from geometry.rotation import Quaternion
from robots.quadrotor import Quadrotor, QuadrotorParams, QuadrotorState


def make_state(
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    quaternion: Quaternion | None = None,
    angular_velocity: np.ndarray | None = None,
) -> QuadrotorState:
    return QuadrotorState(
        position=np.zeros(3) if position is None else position,
        velocity=np.zeros(3) if velocity is None else velocity,
        quaternion=Quaternion.identity() if quaternion is None else quaternion,
        angular_velocity=np.zeros(3) if angular_velocity is None else angular_velocity,
    )


def make_params() -> QuadrotorParams:
    return QuadrotorParams(
        mass=1.0,
        arm_length=0.2,
        inertia=np.diag([0.01, 0.01, 0.02]),
        yaw_torque_coefficient=0.1,
        max_thrust=5.0,
    )


def make_quadrotor(
    state: QuadrotorState | None = None,
    params: QuadrotorParams | None = None,
    thrusts: np.ndarray | None = None,
) -> Quadrotor:
    quadrotor = Quadrotor(
        params=make_params() if params is None else params,
        state=make_state() if state is None else state,
    )
    if thrusts is not None:
        quadrotor.set_thrusts(thrusts)
    return quadrotor
