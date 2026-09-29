"""Base interface for environment disturbances."""

import abc
import numpy as np

from geometry.common.types import FloatVector
from robots.quadrotor.params import QuadrotorParams
from geometry.state import RigidBodyState, RigidBodyStateDerivative


class Disturbance(abc.ABC):
    """Abstract base class for physical disturbances."""

    @abc.abstractmethod
    def apply(
        self,
        state: RigidBodyState,
        thrusts: FloatVector,
        params: QuadrotorParams,
        t: float,
    ) -> tuple[FloatVector, FloatVector]:
        """Compute the disturbance force and torque.
        
        Args:
            state: Current quadrotor state.
            thrusts: Current commanded thrusts in Newtons.
            params: Quadrotor physical parameters.
            t: Simulation time in seconds.
            
        Returns:
            force: Extraneous force vector (3,) in world coordinates [N].
            torque: Extraneous torque vector (3,) in body coordinates [N*m].
        """
        pass
