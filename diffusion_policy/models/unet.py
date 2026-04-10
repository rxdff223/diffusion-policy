"""
Conditional U-Net for Diffusion Policy noise prediction.
Takes noisy action + observation conditioning → predicts noise.
"""
import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(8, out_channels)
        self.group_norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if dimensions differ
        self.skip_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.group_norm1(x).relu()
        h = self.conv1(h)
        h = h + self.time_mlp(self.time_mlp[0](t_emb))[:, :, None]  # add time bias

        h = self.group_norm2(h).relu()
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip_proj(x)


class ConditionalUNet1D(nn.Module):
    """
    Conditional 1D U-Net for action denoising.
    - obs: observation embedding
    - noisy_action: noisy action sequence (T, action_dim)
    - timestep: diffusion timestep
    Returns: predicted noise
    """

    def __init__(
        self,
        obs_dim=1024,
        action_dim=16,
        action_chunk_size=16,
        hidden_dim=256,
        time_emb_dim=128,
        num_blocks=3,
        dropout=0.1,
    ):
        super().__init__()
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Observation encoder: (obs_horizon, obs_dim) -> (hidden_dim,)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: action + time
        self.action_in = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)

        # Time embedding projection for input
        self.time_in = nn.Linear(time_emb_dim, hidden_dim)

        # Encoder blocks (down-sampling via stride)
        self.encoder_blocks = nn.ModuleList()
        ch = hidden_dim
        for i in range(num_blocks):
            self.encoder_blocks.append(ResidualBlock(ch, ch * 2, time_emb_dim, dropout))
            ch *= 2

        # Middle block
        self.middle_block = ResidualBlock(ch, ch, time_emb_dim, dropout)

        # Decoder blocks (up-sampling via transposed conv)
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
                    ResidualBlock(ch // 2, ch // 2, time_emb_dim, dropout),
                )
            )
            ch //= 2

        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1),
        )

    def forward(self, noisy_action, obs, timestep):
        """
        Args:
            noisy_action: (B, action_chunk_size, action_dim)
            obs: (B, obs_horizon, obs_dim) — observation history
            timestep: (B,) — diffusion timestep (0 = clean, T = max noise)
        Returns:
            noise_pred: (B, action_chunk_size, action_dim)
        """
        B = noisy_action.shape[0]

        # Timestep embedding
        t_emb = self.time_mlp(timestep)  # (B, time_emb_dim)

        # Encode observation: mean pool over history -> (B, obs_dim) -> (B, hidden_dim)
        obs_emb = self.obs_encoder(obs).mean(dim=1)  # (B, hidden_dim)
        obs_emb = obs_emb + self.time_in(t_emb)       # (B, hidden_dim)

        # Transpose action for 1D conv: (B, action_dim, T)
        x = noisy_action.transpose(1, 2)  # (B, action_dim, T)

        # Project to hidden dim
        x = self.action_in(x)            # (B, hidden_dim, T)
        x = x + obs_emb[:, :, None]       # add observation conditioning

        # Encoder
        skips = []
        for block in self.encoder_blocks:
            x = block(x, t_emb)
            skips.append(x)
        skips = skips[::-1]  # reverse for decoder

        # Middle
        x = self.middle_block(x, t_emb)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = block[0](x)  # upsample
            x = torch.cat([x, skips[i]], dim=1)  # skip connection
            x = block[1](x, t_emb)  # residual block

        # Output: (B, action_dim, T) -> (B, action_chunk_size, action_dim)
        noise_pred = self.out(x).transpose(1, 2)
        return noise_pred
