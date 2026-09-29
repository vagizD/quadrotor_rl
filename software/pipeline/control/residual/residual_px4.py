"""PX4 Controllers augmented with residual prediction models."""

import collections
import numpy as np
import torch
from pathlib import Path

from geometry.common.types import FloatVector
from robots.quadrotor.params import QuadrotorParams
from geometry.state import RigidBodyState
from software.pipeline.control.px4.px4 import PX4Controller
from research.pipeline.control.residual.models.models import NeuralFlyModel, MLPResidualModel

class BaseResidualPX4Controller(PX4Controller):
    """Base class for residual PX4 controllers with history buffering."""
    
    def __init__(self, params: QuadrotorParams, history_len: int = 10):
        super().__init__(params)
        self.history_len = history_len
        self.state_history = collections.deque(maxlen=history_len)
        self.action_history = collections.deque(maxlen=history_len)
        self.error_history = collections.deque(maxlen=history_len)
        
        # Initialize with zeros
        for _ in range(history_len):
            self.state_history.append(np.zeros(13))
            self.action_history.append(np.zeros(4))
            self.error_history.append(np.zeros(3))
            
        self.last_action = np.zeros(4)

    def _update_history(self, state: RigidBodyState, target_position: FloatVector, action: FloatVector):
        self.state_history.append(state.to_vector())
        self.error_history.append(target_position - state.position)
        self.action_history.append(action)

    def get_history_vector(self) -> np.ndarray:
        s_hist = np.concatenate(self.state_history)
        a_hist = np.concatenate(self.action_history)
        e_hist = np.concatenate(self.error_history)
        return np.concatenate([s_hist, a_hist, e_hist])

    def predict_residual(self, state: RigidBodyState, target_position: FloatVector) -> tuple[FloatVector, FloatVector]:
        """Returns delta_F, delta_tau"""
        return np.zeros(3), np.zeros(3)

    def compute_control(
        self,
        state: RigidBodyState,
        target_position: FloatVector,
        target_yaw: float,
        dt: float,
    ) -> FloatVector:
        
        delta_F, delta_tau = self.predict_residual(state, target_position)
        
        action = super().compute_control(
            state, 
            target_position, 
            target_yaw, 
            dt, 
            delta_F=delta_F, 
            delta_tau=delta_tau
        )
        
        thrusts = action * self.params.max_thrust
        self._update_history(state, target_position, thrusts)
        self.last_action = thrusts
            
        return action

class MLPResidualPX4Controller(BaseResidualPX4Controller):
    """Predicts residual using a trained MLP on history + error."""
    
    def __init__(self, params: QuadrotorParams, model_path: str | Path, history_len: int = 10):
        super().__init__(params, history_len)
        
        # model should take history_len * (13 + 4 + 3) as input
        input_dim = history_len * (13 + 4 + 3)
        self.model = MLPResidualModel(state_dim=input_dim, action_dim=0, wind_dim=0, output_dim=6)
        
        # Adjust state_dict if necessary, but assume model matches architecture
        # We will need to retrain this specific MLP Architecture!
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.model.eval()
        
    def predict_residual(self, state: RigidBodyState, target_position: FloatVector) -> tuple[FloatVector, FloatVector]:
        hist_vec = self.get_history_vector()
        hist_tensor = torch.tensor(hist_vec, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred = self.model(hist_tensor, torch.empty(1, 0), torch.empty(1, 0)).squeeze(0).numpy()
            
        return pred[:3], pred[3:]

class RLResidualPX4Controller(BaseResidualPX4Controller):
    """Predicts residual using an RL policy."""
    
    def __init__(self, params: QuadrotorParams, policy_model, history_len: int = 10):
        super().__init__(params, history_len)
        self.policy = policy_model # Could be a stable-baselines3 model
        
    def predict_residual(self, state: RigidBodyState, target_position: FloatVector) -> tuple[FloatVector, FloatVector]:
        if self.policy is None:
            return np.zeros(3), np.zeros(3)
            
        hist_vec = self.get_history_vector()
        action, _ = self.policy.predict(hist_vec, deterministic=True)
        return action[:3], action[3:]

class NeuralFlyPX4Controller(BaseResidualPX4Controller):
    """Augments the deterministic PX4 controller with a Neural-Fly residual."""
    
    def __init__(self, params: QuadrotorParams, model_path: str | Path, num_basis: int = 16, lr: float = 0.01, a_max: float = 5.0):
        super().__init__(params, history_len=1) # Doesn't need long history
        
        self.model = NeuralFlyModel(state_dim=13, action_dim=4, wind_dim=3, num_basis=num_basis)
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.model.eval()
        
        self.num_basis = num_basis
        self.a_hat = np.zeros(num_basis)
        self.lr = lr
        self.a_max = a_max
        self.dt = 0.01

    def predict_residual(self, state: RigidBodyState, target_position: FloatVector) -> tuple[FloatVector, FloatVector]:
        s_vec = state.to_vector()
        
        state_tensor = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(self.last_action, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            phi = self.model.basis_net(state_tensor, action_tensor).squeeze(0).numpy()
            
        phi_f = phi[:3, :]
        phi_tau = phi[3:, :]
        
        pred_force = phi_f @ self.a_hat
        pred_torque = phi_tau @ self.a_hat
        
        e_p = target_position - state.position
        e_v = -state.velocity 
        s = e_v + 2.0 * e_p
        
        sigma = 0.1
        update = -self.lr * (phi_f.T @ s) - sigma * self.a_hat
        
        self.a_hat += update * self.dt
        self.a_hat = np.clip(self.a_hat, -self.a_max, self.a_max)
        
        return pred_force, pred_torque
