"""
Diffusion Process: DDPM-based conditional diffusion for robot actions.
Implements forward noising, reverse denoising, and training/ inference loops.
"""
import torch
import torch.nn as nn
import numpy as np


def get_noise_schedule(schedule_type, num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Create noise schedule (betas) for the forward diffusion process."""
    if schedule_type == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
    elif schedule_type == "cosine":
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")
    return betas


class DiffusionProcess(nn.Module):
    """
    DDPM-based Diffusion Process for action sequences.
    Training: predict noise given noisy action + conditioning.
    Inference: iterative denoising from pure noise.
    """

    def __init__(
        self,
        noise_predictor,
        num_timesteps=100,
        beta_schedule="cosine",
        device="cuda",
    ):
        super().__init__()
        self.noise_predictor = noise_predictor
        self.num_timesteps = num_timesteps
        self.device = device

        # Noise schedule
        betas = get_noise_schedule(beta_schedule, num_timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

        # Register as buffers (moved to device automatically)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward_diffusion(self, x0, timesteps):
        """
        Add noise to clean action x0 at given timesteps.
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        Args:
            x0: (B, action_chunk_size, action_dim) — clean actions
            timesteps: (B,) — noise level per sample
        Returns:
            x_t: noisy actions
            noise: the added noise (for loss computation)
        """
        noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    def p_mean_variance(self, x_t, obs, timesteps):
        """
        Compute mean and variance of p_theta(x_{t-1} | x_t, c).
        Returns the parameters of the reverse Gaussian.
        """
        noise_pred = self.noise_predictor(x_t, obs, timesteps)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[timesteps][:, None, None]
        sqrt_one_minus  = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        alpha_bar_prev  = self.alphas_cumprod_prev[timesteps][:, None, None]
        beta_t          = self.betas[timesteps][:, None, None]

        x_recon = (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha_bar.clamp(min=1e-8)
        model_mean = alpha_bar_prev.sqrt() * x_recon + beta_t.sqrt() * noise_pred
        model_var   = beta_t
        return model_mean, model_var, noise_pred

    @torch.no_grad()
    def reverse_step(self, x_t, obs, timesteps, sampling_strategy="ddpm"):
        """
        Single reverse diffusion step.
        Args:
            x_t: current noisy action
            obs: conditioning observations
            timesteps: current timesteps (B,)
            sampling_strategy: 'ddpm' (standard) or 'ddim' (faster)
        Returns:
            x_{t-1}: denoised action
        """
        B = x_t.shape[0]

        if sampling_strategy == "ddpm":
            # Standard DDPM sampling
            noise_pred = self.noise_predictor(x_t, obs, timesteps)
            alpha_t    = self.alphas[timesteps][:, None, None]          # (B, 1, 1)
            alpha_bar_t = self.alphas_cumprod[timesteps][:, None, None]
            beta_t     = self.betas[timesteps][:, None, None]

            # Predicted clean action
            x_recon = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t).clamp(min=1e-8)

            # Mean of reverse distribution
            mean = (torch.sqrt(alpha_bar_t) * beta_t / (1 - alpha_bar_t + 1e-8)) * x_recon + \
                   (torch.sqrt(alpha_t) * (1 - alpha_bar_t) / (1 - alpha_bar_t + 1e-8)) * x_t

            # Sample from N(mean, var)
            if timesteps[0] == 0:
                x_prev = mean
            else:
                var = beta_t
                x_prev = mean + torch.sqrt(var) * torch.randn_like(x_t)

        elif sampling_strategy == "ddim":
            # DDIM sampling (deterministic or partial stochastic)
            noise_pred = self.noise_predictor(x_t, obs, timesteps)
            alpha_bar_t    = self.alphas_cumprod[timesteps][:, None, None]
            x_recon = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t).clamp(min=1e-8)

            # DDIM "eta=0" gives deterministic reverse
            alpha_bar_prev = self.alphas_cumprod_prev[timesteps][:, None, None]
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
            x_prev = torch.sqrt(alpha_bar_prev) * x_recon + dir_xt

        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

        return x_prev

    @torch.no_grad()
    def sample(self, obs, sampling_strategy="ddpm", num_steps=None):
        """
        Generate action from pure noise given observations.
        Args:
            obs: (B, obs_horizon, obs_dim) — observation history
            sampling_strategy: 'ddpm' or 'ddim'
            num_steps: for DDIM, fewer steps than full T (e.g. 10-20)
        Returns:
            x0: (B, action_chunk_size, action_dim) — denoised action
        """
        if sampling_strategy == "ddim":
            return self._sample_ddim(obs, num_steps)
        else:
            return self._sample_ddpm(obs)

    def _sample_ddpm(self, obs):
        """Standard DDPM sampling (full T steps)."""
        B = obs.shape[0]
        device = self.device

        # Start from pure Gaussian noise
        x_t = torch.randn(B, self.noise_predictor.action_chunk_size,
                          self.noise_predictor.action_dim, device=device)

        for t in reversed(range(self.num_timesteps)):
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            x_t = self.reverse_step(x_t, obs, timesteps, sampling_strategy="ddpm")

        return x_t

    def _sample_ddim(self, obs, num_steps=None):
        """DDIM sampling with fewer steps (faster)."""
        if num_steps is None:
            num_steps = self.num_timesteps

        B = obs.shape[0]
        device = self.device

        # Create step schedule
        step_list = np.linspace(0, self.num_timesteps - 1, num_steps).astype(int).tolist()
        step_list = sorted(set(step_list))  # deduplicate

        # Start from pure noise
        x_t = torch.randn(B, self.noise_predictor.action_chunk_size,
                          self.noise_predictor.action_dim, device=device)

        for i, t in enumerate(reversed(step_list)):
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            x_t = self.reverse_step(x_t, obs, timesteps, sampling_strategy="ddim")

        return x_t

    def compute_loss(self, x0, obs):
        """
        Compute DDPM L2 loss.
        Args:
            x0: (B, action_chunk_size, action_dim) — clean action targets
            obs: (B, obs_horizon, obs_dim) — observation history
        Returns:
            loss: scalar MSE between predicted and true noise
        """
        B = x0.shape[0]
        device = x0.device

        # Sample random timesteps
        timesteps = torch.randint(0, self.num_timesteps, (B,), device=device).long()

        # Forward: add noise
        x_t, noise = self.forward_diffusion(x0, timesteps)

        # Predict noise
        noise_pred = self.noise_predictor(x_t, obs, timesteps)

        # L2 loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss
