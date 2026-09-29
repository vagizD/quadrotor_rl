import numpy as np
from typing import Callable

from geometry.state import RigidBodyState
from geometry.math.rotation import Quaternion
from research.evaluation.quadrotor.scenarios.base import Scenario
from research.simulator.simulator import QuadrotorEnvironment

class PointToPointScenario(Scenario):
    """Fly from origin to a specific target."""
    
    def __init__(self, env: QuadrotorEnvironment, duration: float = 10.0, dt: float = 0.01):
        super().__init__("point_to_point", duration, dt, env)

    def get_initial_state(self) -> RigidBodyState:
        return RigidBodyState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=Quaternion.identity(), angular_velocity=np.zeros(3)
        )

    def get_target_trajectory(self) -> Callable[[float], np.ndarray]:
        return lambda t: np.array([5.0, -3.0, 4.0])
