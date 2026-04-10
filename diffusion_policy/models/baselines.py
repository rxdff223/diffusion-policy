"""
Behavioral Cloning and DAgger baselines for comparison.
Student 3 contributes these as RL/comparison baselines.
"""
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """
    Simple Behavioral Cloning policy.
    MLP that maps observation → action.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256, obs_horizon=2, dropout=0.1):
        super().__init__()
        self.obs_horizon = obs_horizon
        # Flatten observation history
        self.net = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, obs):
        """
        Args:
            obs: (B, obs_horizon, obs_dim)
        Returns:
            action: (B, action_dim)
        """
        x = obs.flatten(1)  # (B, obs_horizon * obs_dim)
        return self.net(x)

    def compute_loss(self, obs, action_target):
        """MSE loss for BC."""
        action_pred = self.forward(obs)  # (B, action_dim)
        return nn.functional.mse_loss(action_pred, action_target)


class DAggerPolicy(nn.Module):
    """
    DAgger policy (same architecture as BC, trained with iterative dataset aggregation).
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256, obs_horizon=2, dropout=0.1):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.net = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, obs):
        x = obs.flatten(1)
        return self.net(x)
