import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class BasisNetwork(nn.Module):
    def __init__(self, state_dim=13, action_dim=4, num_basis=16, output_dim=6):
        super().__init__()
        self.num_basis = num_basis
        self.output_dim = output_dim
        
        # phi(x, u) -> Matrix of shape (output_dim, num_basis)
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(state_dim + action_dim, 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, output_dim * num_basis))
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        # Reshape to (batch_size, output_dim, num_basis)
        return out.view(-1, self.output_dim, self.num_basis)

class WindEncoder(nn.Module):
    def __init__(self, wind_dim=3, num_basis=16):
        super().__init__()
        # a(w) -> vector of shape (num_basis)
        self.net = nn.Sequential(
            nn.Linear(wind_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_basis)
        )
        
    def forward(self, w):
        return self.net(w)

class NeuralFlyModel(nn.Module):
    def __init__(self, state_dim=13, action_dim=4, wind_dim=3, num_basis=16, output_dim=6):
        super().__init__()
        self.basis_net = BasisNetwork(state_dim, action_dim, num_basis, output_dim)
        self.wind_encoder = WindEncoder(wind_dim, num_basis)
        
    def forward(self, state, action, wind):
        # phi shape: (B, output_dim, num_basis)
        phi = self.basis_net(state, action)
        # a shape: (B, num_basis)
        a = self.wind_encoder(wind)
        
        # phi * a: (B, output_dim, num_basis) @ (B, num_basis, 1) -> (B, output_dim, 1)
        a = a.unsqueeze(-1)
        out = torch.bmm(phi, a).squeeze(-1)
        return out

class MLPResidualModel(nn.Module):
    """A simple baseline MLP that predicts the disturbance directly from state, action, and wind."""
    def __init__(self, state_dim=13, action_dim=4, wind_dim=3, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(state_dim + action_dim + wind_dim, 128)),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, output_dim))
        )
        
    def forward(self, state, action, wind):
        x = torch.cat([state, action, wind], dim=-1)
        return self.net(x)
