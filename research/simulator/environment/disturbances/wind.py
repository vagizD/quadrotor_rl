"""Wind force disturbance."""

import numpy as np

from geometry.common.types import FloatVector
from robots.quadrotor.params import QuadrotorParams
from geometry.state import RigidBodyState, RigidBodyStateDerivative
from .base import Disturbance

class WindForce(Disturbance):
    """Applies a sophisticated wind force with constant base and stochastic gusts."""
    
    def __init__(self, 
                 constant_force: FloatVector = np.zeros(3), 
                 gust_amplitude: float = 0.0,
                 gust_frequencies: list[float] = None,
                 noise_std: float = 0.0):
        self.base_force = np.array(constant_force, dtype=np.float64)
        self.gust_amplitude = gust_amplitude
        self.gust_frequencies = gust_frequencies or [0.5, 1.2, 2.7]
        self.noise_std = noise_std
        
    def apply(
        self,
        state: RigidBodyState,
        thrusts: FloatVector,
        params: QuadrotorParams,
        t: float,
    ) -> tuple[FloatVector, FloatVector]:
        
        force = self.base_force.copy()
        
        if self.gust_amplitude > 0:
            # Complex gust using sum of sines with different frequencies
            gust_x = sum(np.sin(freq * t) for freq in self.gust_frequencies)
            gust_y = sum(np.cos(freq * t * 1.1) for freq in self.gust_frequencies)
            gust_z = sum(np.sin(freq * t * 0.9) for freq in self.gust_frequencies)
            
            # Normalize sum of sines to approximately [-1, 1] then scale
            normalization = len(self.gust_frequencies)
            force += (self.gust_amplitude / normalization) * np.array([gust_x, gust_y, gust_z], dtype=np.float64)
            
        if self.noise_std > 0:
            force += np.random.normal(0, self.noise_std, size=3)
            
        torque = np.zeros(3, dtype=np.float64)
        return force, torque
