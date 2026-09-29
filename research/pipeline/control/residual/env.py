import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robots.quadrotor.params import QuadrotorParams
from geometry.state import RigidBodyState
from geometry.math.rotation import Quaternion
from software.pipeline.control.px4.px4 import PX4Controller
from research.simulator.simulator import QuadrotorEnvironment
from research.simulator.integrators.rk4 import RK4Integrator
from research.simulator.environment.disturbances import WindForce

class ResidualLearningEnv(gym.Env):
    """
    Gym environment for training an RL agent to predict residual forces/torques.
    The agent sees history of states, actions, and tracking errors.
    The underlying PX4 controller computes the base action, which is augmented
    by the RL agent's predicted delta_F and delta_tau.
    """
    def __init__(self, history_len: int = 10, max_steps: int = 500):
        super().__init__()
        
        self.history_len = history_len
        self.max_steps = max_steps
        
        self.params = QuadrotorParams(
            mass=1.0,
            arm_length=0.25,
            inertia=np.diag([0.02, 0.02, 0.04]),
            yaw_torque_coefficient=0.01,
            max_thrust=20.0,
            gravity=9.81,
            motor_spin_directions=np.array([1, -1, 1, -1])
        )
        self.integrator = RK4Integrator(dt=0.01)
        self.controller = PX4Controller(self.params)
        
        # Action is delta_F (3) and delta_tau (3)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(6,), dtype=np.float32)
        
        # Observation is history of (state(13), action(4), error(3))
        obs_dim = history_len * (13 + 4 + 3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Domain Randomization: Wind
        wind_type = np.random.randint(0, 4)
        if wind_type == 0:
            wind = np.zeros(3)
        elif wind_type == 1:
            wind = np.random.uniform(-2.0, 2.0, size=3)
        elif wind_type == 2:
            wind = np.random.uniform(-5.0, 5.0, size=3)
        else:
            wind = np.random.uniform(-10.0, 10.0, size=3)
            
        disturbance = WindForce(constant_force=wind.tolist())
        
        initial_state = RigidBodyState(
            position=np.random.uniform(-2, 2, size=3),
            velocity=np.random.uniform(-1, 1, size=3),
            quaternion=Quaternion.identity(),
            angular_velocity=np.random.uniform(-0.5, 0.5, size=3)
        )
        
        self.env = QuadrotorEnvironment(
            params=self.params,
            integrator=self.integrator,
            initial_state=initial_state,
            disturbances=[disturbance]
        )
        
        # Random target position
        self.target_position = np.random.uniform(-5, 5, size=3)
        self.target_yaw = 0.0
        
        # Initialize history
        self.state_history = [np.zeros(13) for _ in range(self.history_len)]
        self.action_history = [np.zeros(4) for _ in range(self.history_len)]
        self.error_history = [np.zeros(3) for _ in range(self.history_len)]
        
        self.step_count = 0
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        s_hist = np.concatenate(self.state_history)
        a_hist = np.concatenate(self.action_history)
        e_hist = np.concatenate(self.error_history)
        return np.concatenate([s_hist, a_hist, e_hist]).astype(np.float32)
        
    def step(self, action):
        delta_F = action[:3]
        delta_tau = action[3:]
        
        # Base control + residual
        base_action = self.controller.compute_control(
            self.env.quadrotor.state,
            self.target_position,
            self.target_yaw,
            0.01,
            delta_F=delta_F,
            delta_tau=delta_tau
        )
        
        thrusts = base_action * self.params.max_thrust
        self.env.step(thrusts)
        
        self.step_count += 1
        
        # Update history
        state_vec = self.env.quadrotor.state.to_vector()
        error = self.target_position - self.env.quadrotor.state.position
        
        self.state_history.pop(0)
        self.state_history.append(state_vec)
        
        self.action_history.pop(0)
        self.action_history.append(thrusts)
        
        self.error_history.pop(0)
        self.error_history.append(error)
        
        # Reward inversely proportional to position error
        error_norm = np.linalg.norm(error)
        reward = -error_norm
        
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(), float(reward), terminated, truncated, {}
