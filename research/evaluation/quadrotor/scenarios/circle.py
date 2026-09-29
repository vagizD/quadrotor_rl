import numpy as np
from typing import Callable

from geometry.state import RigidBodyState
from geometry.math.rotation import Quaternion
from research.evaluation.quadrotor.scenarios.base import Scenario
from research.simulator.simulator import QuadrotorEnvironment

class CircleScenario(Scenario):
    """Track a circular trajectory."""
    
    def __init__(self, env: QuadrotorEnvironment, duration: float = 20.0, dt: float = 0.01):
        super().__init__("circle", duration, dt, env)

    def get_initial_state(self) -> RigidBodyState:
        return RigidBodyState(
            position=np.array([2.0, 0.0, 2.0]),
            velocity=np.zeros(3),
            quaternion=Quaternion.identity(), angular_velocity=np.zeros(3)
        )

    def get_target_trajectory(self) -> Callable[[float], np.ndarray]:
        omega = 1.0  # rad/s
        return lambda t: np.array([2.0 * np.cos(omega * t), 2.0 * np.sin(omega * t), 2.0])
