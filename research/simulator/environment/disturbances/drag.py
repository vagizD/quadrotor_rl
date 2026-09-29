"""Aerodynamic drag disturbance."""

import numpy as np

from geometry.common.types import FloatVector
from robots.quadrotor.params import QuadrotorParams
from geometry.state import RigidBodyState, RigidBodyStateDerivative
from .base import Disturbance

class AerodynamicDrag(Disturbance):
    """Quadratic aerodynamic drag on the quadrotor body."""
    
    def __init__(self, linear_drag_coeff: float = 0.1, angular_drag_coeff: float = 0.01):
        self.c_v = linear_drag_coeff
        self.c_w = angular_drag_coeff
        
    def apply(
        self,
        state: RigidBodyState,
        thrusts: FloatVector,
        params: QuadrotorParams,
        t: float,
    ) -> tuple[FloatVector, FloatVector]:
        
        v_norm = np.linalg.norm(state.velocity)
        force = -self.c_v * v_norm * state.velocity
        
        w_norm = np.linalg.norm(state.angular_velocity)
        torque = -self.c_w * w_norm * state.angular_velocity
        
        return force, torque
