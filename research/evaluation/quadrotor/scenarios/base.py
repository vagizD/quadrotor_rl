import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable

from robots.quadrotor.model import Quadrotor
from research.simulator.simulator import QuadrotorEnvironment
from typing import Any
from research.pipeline.planning.trajectory import Trajectory
from geometry.state import RigidBodyState
from geometry.math.rotation import Quaternion


class Scenario(ABC):
    """Base class for all quadrotor evaluation scenarios."""

    def __init__(
        self,
        name: str,
        duration: float,
        dt: float,
        env: QuadrotorEnvironment,
    ):
        self.name = name
        self.duration = duration
        self.dt = dt
        self.env = env

    @abstractmethod
    def get_initial_state(self) -> RigidBodyState:
        """Return the initial state of the quadrotor."""
        pass

    @abstractmethod
    def get_target_trajectory(self) -> Callable[[float], np.ndarray]:
        """Return a function that gives the target position at time t."""
        pass

    def run(self, controller: Any) -> Trajectory:
        """Run the scenario and return the resulting trajectory."""
        state = self.get_initial_state()
        self.env.reset(state)

        target_fn = self.get_target_trajectory()

        states = []
        target_positions = []
        actions = []
        times = []

        num_steps = int(self.duration / self.dt)
        for _ in range(num_steps):
            t = self.env.time
            current_target = target_fn(t)

            states.append(self.env.quadrotor.state.to_vector())
            target_positions.append(current_target)
            times.append(t)

            action = controller.compute_control(self.env.quadrotor.state, current_target, 0.0, self.dt)
            thrusts = action * self.env.params.max_thrust
            actions.append(thrusts)

            self.env.step(thrusts)
            
        # Append the final state
        t = self.env.time
        states.append(self.env.quadrotor.state.to_vector())
        target_positions.append(target_fn(t))
        times.append(t)

        return Trajectory(
            states=np.array(states),
            target_positions=np.array(target_positions),
            thrusts=np.array(actions),
            times=np.array(times),
        )
