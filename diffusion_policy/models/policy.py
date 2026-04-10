"""
Diffusion Policy: combines U-Net noise predictor + DDPM diffusion process.
Entry point for training and inference.
"""
import torch
import torch.nn as nn
from .unet import ConditionalUNet1D
from .diffusion import DiffusionProcess


class DiffusionPolicy(nn.Module):
    """
    Full Diffusion Policy model.
    - observation_encoder: processes obs history
    - noise_predictor: U-Net that predicts noise given noisy action + conditioning
    - diffusion: handles forward/backward process, sampling
    """

    def __init__(self, obs_dim, action_dim, action_chunk_size=16, obs_horizon=2,
                 hidden_dim=256, time_emb_dim=128, num_blocks=3, dropout=0.1,
                 num_timesteps=100, beta_schedule="cosine", device="cuda"):
        super().__init__()
        self.device = device

        # U-Net noise predictor
        self.noise_predictor = ConditionalUNet1D(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_blocks=num_blocks,
            dropout=dropout,
        )

        # Diffusion process (wraps the predictor)
        self.diffusion = DiffusionProcess(
            noise_predictor=self.noise_predictor,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            device=device,
        )

        # Store dims for convenience
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.obs_horizon = obs_horizon

    def forward(self, x0, obs):
        """Compute training loss."""
        return self.diffusion.compute_loss(x0, obs)

    @torch.no_grad()
    def sample_actions(self, obs, sampling_strategy="ddpm", num_steps=None):
        """
        Generate action given current observations.
        Args:
            obs: (B, obs_horizon, obs_dim) or (obs_horizon, obs_dim)
            sampling_strategy: 'ddpm' or 'ddim'
            num_steps: for DDIM, number of denoising steps
        Returns:
            actions: (B, action_chunk_size, action_dim) or (action_chunk_size, action_dim)
        """
        # Handle single-sample case (no batch dim)
        single = obs.dim() == 2
        if single:
            obs = obs.unsqueeze(0)

        obs = obs.to(self.device)

        # Sample from diffusion model
        action = self.diffusion.sample(obs, sampling_strategy, num_steps)

        if single:
            action = action.squeeze(0)

        return action

    def to(self, device):
        self.device = device
        return super().to(device)
