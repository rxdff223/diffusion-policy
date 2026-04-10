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
        # GN norm is applied to the actual input channels
        self.group_norm1 = nn.GroupNorm(8, in_channels)
        self.group_norm2 = nn.GroupNorm(8, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.dropout = nn.Dropout(dropout)

        # Skip connection: project if dimensions differ
        self.skip_proj = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.group_norm1(x).relu()
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None]

        h = self.group_norm2(h).relu()
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip_proj(x)


class ConditionalUNet1D(nn.Module):
    """
    Conditional 1D U-Net for action denoising.
    - noisy_action: noisy action sequence (B, action_chunk_size, action_dim)
    - obs: observation conditioning (B, obs_horizon, obs_dim)
    - timestep: diffusion timestep (B,)
    Returns: predicted noise (B, action_chunk_size, action_dim)
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

        # Observation encoder: per-frame embedding + bidirectional LSTM → hidden_dim
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.obs_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.obs_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # bidirectional → hidden_dim
        self.obs_to_time = nn.Linear(hidden_dim, time_emb_dim)    # hidden_dim → time_emb_dim

        # Input: (B, action_dim, T) — action is transposed for 1D conv
        self.action_in = nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1)

        # Build encoder: each block doubles channels, halves sequence length
        self.encoder_convs = nn.ModuleList()
        self.encoder_res = nn.ModuleList()
        ch = hidden_dim
        for i in range(num_blocks):
            self.encoder_convs.append(nn.Conv1d(ch, ch * 2, kernel_size=3, stride=2, padding=1))
            self.encoder_res.append(ResidualBlock(ch * 2, ch * 2, time_emb_dim, dropout))
            ch *= 2

        # Middle block
        self.middle = ResidualBlock(ch, ch, time_emb_dim, dropout)

        # Build decoder: each block halves channels, doubles sequence length
        self.decoder_convs = nn.ModuleList()
        self.decoder_res = nn.ModuleList()
        for i in range(num_blocks):
            self.decoder_convs.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1)
            )
            self.decoder_res.append(ResidualBlock(ch, ch // 2, time_emb_dim, dropout))
            ch //= 2

        # Output: predict noise
        self.out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1),
        )

    def forward(self, noisy_action, obs, timestep):
        """
        Args:
            noisy_action: (B, action_chunk_size, action_dim)
            obs: (B, obs_horizon, obs_dim)
            timestep: (B,)
        Returns:
            noise_pred: (B, action_chunk_size, action_dim)
        """
        B = noisy_action.shape[0]

        # Timestep embedding
        t_emb = self.time_mlp(timestep)  # (B, time_emb_dim)

        # Encode observation: (B, obs_horizon, obs_dim)
        # 1) per-frame embedding
        obs_emb = self.obs_embed(obs)                # (B, obs_horizon, hidden_dim)
        # 2) bidirectional LSTM → final hidden state
        _, (h_n, _) = self.obs_lstm(obs_emb)        # h_n: (2, B, hidden_dim)
        obs_emb = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, hidden_dim*2), forward+backward
        obs_emb = self.obs_proj(obs_emb)             # (B, hidden_dim)
        obs_t_emb = self.obs_to_time(obs_emb)        # (B, time_emb_dim)
        t_emb = t_emb + obs_t_emb                   # combine time + obs conditioning

        # Transpose action for 1D conv: (B, action_dim, T)
        x = noisy_action.transpose(1, 2)

        # Project to hidden dim
        x = self.action_in(x)  # (B, hidden_dim, T)

        # Encoder: down-sample + residual
        skips = []
        for conv, res in zip(self.encoder_convs, self.encoder_res):
            skips.append(x)            # store BEFORE downsample (matches upsampled length)
            x = conv(x)               # downsample, double channels
            x = res(x, t_emb)         # residual block

        # Middle
        x = self.middle(x, t_emb)

        # Decoder: up-sample + concatenate skip + residual
        for conv, res, skip_ch in zip(
            self.decoder_convs, self.decoder_res, reversed(skips)
        ):
            x = conv(x)                   # upsample, half channels → ch//2
            x = torch.cat([x, skip_ch], dim=1)  # concat → ch channels
            x = res(x, t_emb)             # residual: ch → ch//2

        # Output: (B, action_dim, T) → (B, action_chunk_size, action_dim)
        noise_pred = self.out(x).transpose(1, 2)
        return noise_pred
