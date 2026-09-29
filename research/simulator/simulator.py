import numpy as np

from geometry.common.types import FloatVector
from robots.quadrotor import Quadrotor, QuadrotorParams, RigidBodyState, compute_state_derivative_vector
from research.simulator.integrators import EulerIntegrator, RK4Integrator
from research.simulator.environment.disturbances.base import Disturbance
from research.simulator.environment.obstacles import PhysicalEnvironment

class QuadrotorEnvironment:
    """Core physics environment for the quadrotor with environmental disturbances."""

    def __init__(
        self,
        params: QuadrotorParams,
        integrator: EulerIntegrator | RK4Integrator,
        initial_state: RigidBodyState,
        disturbances: list[Disturbance] | None = None,
        physical_env: PhysicalEnvironment | None = None,
    ) -> None:
        if not isinstance(params, QuadrotorParams):
            raise TypeError("params must be QuadrotorParams")
        if not isinstance(integrator, (EulerIntegrator, RK4Integrator)):
            raise TypeError("integrator must be EulerIntegrator or RK4Integrator")
        if not isinstance(initial_state, RigidBodyState):
            raise TypeError("initial_state must be RigidBodyState")

        self.params = params
        self.integrator = integrator
        self.disturbances = disturbances or []
        self.physical_env = physical_env
        self.collision_info = (False, None)
        self.quadrotor = Quadrotor(params, initial_state)
        self.time = 0.0

    @property
    def state(self) -> RigidBodyState:
        return self.quadrotor.state

    def reset(self, initial_state: RigidBodyState) -> None:
        if not isinstance(initial_state, RigidBodyState):
            raise TypeError("initial_state must be RigidBodyState")
        self.quadrotor.state = RigidBodyState(
            position=initial_state.position,
            velocity=initial_state.velocity,
            quaternion=initial_state.quaternion,
            angular_velocity=initial_state.angular_velocity,
        )
        self.time = 0.0

    def step(self, thrusts: FloatVector) -> None:
        """Advance the physics simulation by one integration step."""
        self.quadrotor.set_thrusts(thrusts)
        current_thrusts = self.quadrotor.thrusts.copy()
        state_vector = self.state.to_vector()

        def derivative_function(current: FloatVector) -> FloatVector:
            current_state = RigidBodyState.from_vector(current)
            ext_f = np.zeros(3, dtype=np.float64)
            ext_tq = np.zeros(3, dtype=np.float64)
            
            for d in self.disturbances:
                f, tq = d.apply(current_state, current_thrusts, self.params, self.time)
                ext_f += f
                ext_tq += tq
                
            return compute_state_derivative_vector(
                current, current_thrusts, self.params, external_force=ext_f, external_torque=ext_tq
            )

        next_state_vector = self.integrator.step(state_vector, derivative_function)
        next_state = RigidBodyState.from_vector(next_state_vector)
        next_state.quaternion = next_state.quaternion.normalized()
        
        # Stop integration/movement if collision occurs and register it
        if self.physical_env is not None:
            # We use arm_length roughly as the collision radius
            is_collision, source = self.physical_env.check_collision(
                next_state.position, radius=self.params.arm_length
            )
            if is_collision:
                self.collision_info = (True, source)
                # Simple inelastic stop: keep previous position/velocity if penetrated
                next_state.position = self.state.position
                next_state.velocity = np.zeros(3)
                next_state.angular_velocity = np.zeros(3)

        self.quadrotor.state = next_state
        self.time += self.integrator.dt
