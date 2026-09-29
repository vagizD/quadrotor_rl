import numpy as np

from geometry.common.types import FloatVector
from robots.quadrotor.params import QuadrotorParams
from robots.quadrotor.motors import compute_total_thrust
from geometry.state import RigidBodyState
from .base import Disturbance


class GroundEffect(Disturbance):
    """Ground effect model increasing vertical thrust near the surface."""

    def __init__(self, ground_z: float = 0.0, cutoff_distance: float = 0.5, factor: float = 0.5):
        self.ground_z = ground_z
        self.cutoff_distance = cutoff_distance
        self.factor = factor

    def apply(
        self,
        state: RigidBodyState,
        thrusts: FloatVector,
        params: QuadrotorParams,
        t: float,
    ) -> tuple[FloatVector, FloatVector]:
        h = state.position[2] - self.ground_z
        
        if h > self.cutoff_distance or h < 0.0:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        total_thrust = compute_total_thrust(thrusts)
        
        # F_ground = k(h)T, where k(h) is a non-linear scaling factor depending on distance h
        k_h = self.factor * (1.0 - h / self.cutoff_distance)**2
        
        # Ground effect pushes upwards in World Z
        f_ground = np.array([0.0, 0.0, k_h * total_thrust], dtype=np.float64)
        
        return f_ground, np.zeros(3, dtype=np.float64)
