"""
Data loader for Diffusion Policy.
Handles normalization, observation history window, and action chunking.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class RobotDataset(Dataset):
    def __init__(self, data_dir, action_chunk_size=16, obs_horizon=2, normalizer=None):
        self.data_dir = data_dir
        self.action_chunk_size = action_chunk_size
        self.obs_horizon = obs_horizon

        # Load preprocessed data
        self.actions = np.load(f"{data_dir}/liftpot_actions.npy")
        self.observations = np.load(f"{data_dir}/liftpot_images.npy")

        with open(f"{data_dir}/stats.json") as f:
            self.stats = json.load(f)

        self.action_min = np.array(self.stats["action_min"])
        self.action_max = np.array(self.stats["action_max"])

        # Number of valid starting positions per trajectory
        self.n_demos, self.n_timesteps, self.action_dim = self.actions.shape
        self.obs_dim = self.observations.shape[2]

        # For action chunking, we need at least obs_horizon + action_chunk_size
        self.valid_starts = self.n_timesteps - obs_horizon - action_chunk_size + 1

        self.normalizer = normalizer or self._create_normalizer()

    def _create_normalizer(self):
        """Create a normalizer from the stats."""
        action_min = torch.tensor(self.action_min, dtype=torch.float32)
        action_max = torch.tensor(self.action_max, dtype=torch.float32)
        return ActionNormalizer(action_min, action_max)

    def __len__(self):
        # Sample random (demo, start_t) pairs
        return self.n_demos * self.valid_starts

    def __getitem__(self, idx):
        # Map flat idx to (demo_id, start_t)
        demo_id = idx // self.valid_starts
        start_t = idx % self.valid_starts

        # Observation history: last obs_horizon frames
        obs_start = start_t
        obs_end = start_t + self.obs_horizon
        obs_history = self.observations[demo_id, obs_start:obs_end]  # (obs_horizon, obs_dim)

        # Target action chunk (action_chunk_size future actions)
        action_start = obs_end
        action_end = action_start + self.action_chunk_size
        action_chunk = self.actions[demo_id, action_start:action_end]  # (action_chunk_size, action_dim)

        # Normalize action to [-1, 1]
        action_chunk_norm = self.normalizer.normalize(action_chunk)

        # Convert to torch tensors
        obs_history = torch.tensor(obs_history, dtype=torch.float32)
        action_chunk = torch.tensor(action_chunk_norm, dtype=torch.float32)

        return {
            "obs_history": obs_history,    # (obs_horizon, obs_dim)
            "action_chunk": action_chunk,    # (action_chunk_size, action_dim)
        }


class ActionNormalizer:
    """Normalizes actions from [min, max] to [-1, 1]."""

    def __init__(self, action_min, action_max):
        # Store as numpy for consistent dtype with dataset output
        self.action_min = np.array(action_min, dtype=np.float32)
        self.action_max = np.array(action_max, dtype=np.float32)
        self.action_range = self.action_max - self.action_min

    def normalize(self, actions):
        """Map [min, max] -> [-1, 1]"""
        return 2.0 * (actions - self.action_min) / self.action_range - 1.0

    def denormalize(self, actions_norm):
        """Map [-1, 1] -> [min, max]"""
        return (actions_norm + 1.0) / 2.0 * self.action_range + self.action_min


def get_dataloader(data_dir, batch_size=32, action_chunk_size=16, obs_horizon=2, shuffle=True):
    dataset = RobotDataset(data_dir, action_chunk_size=action_chunk_size, obs_horizon=obs_horizon)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
    return loader, dataset.normalizer
