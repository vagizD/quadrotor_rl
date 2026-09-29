import numpy as np
from typing import Callable

from geometry.state import RigidBodyState
from geometry.math.rotation import Quaternion
from research.evaluation.quadrotor.scenarios.base import Scenario
from research.simulator.simulator import QuadrotorEnvironment

class HoverScenario(Scenario):
    """Hover exactly at the target position."""
    
    def __init__(self, env: QuadrotorEnvironment, duration: float = 10.0, dt: float = 0.01):
        super().__init__("hover", duration, dt, env)

    def get_initial_state(self) -> RigidBodyState:
        # Start at [2.0, 2.0, 0.0]
        return RigidBodyState(
            position=np.array([2.0, 2.0, 0.0]),
            velocity=np.zeros(3),
            quaternion=Quaternion.identity(), angular_velocity=np.zeros(3)
        )

    def get_target_trajectory(self) -> Callable[[float], np.ndarray]:
        return lambda t: np.array([0.0, 0.0, 2.0])
