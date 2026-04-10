"""
Quick test script to verify the full pipeline works before running on server.
Run: python scripts/test_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
from diffusion_policy.models.policy import DiffusionPolicy
from diffusion_policy.utils.dataset import RobotDataset, get_dataloader


def main():
    print("=" * 50)
    print("Diffusion Policy Pipeline Test")
    print("=" * 50)

    device = "cpu"

    # ── Test 1: Dataset ──────────────────────────────────────────
    print("\n=== Test 1: Dataset ===")
    dataset = RobotDataset("preprocessed", action_chunk_size=16, obs_horizon=2)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  obs_dim: {dataset.obs_dim}, action_dim: {dataset.action_dim}")

    sample = dataset[0]
    assert sample["obs_history"].shape == (2, 1024)
    assert sample["action_chunk"].shape == (16, 16)
    assert -1.0 <= sample["action_chunk"].min() <= sample["action_chunk"].max() <= 1.0
    print("  ✓ Dataset OK")

    # ── Test 2: DataLoader ────────────────────────────────────────
    print("\n=== Test 2: DataLoader ===")
    loader, normalizer = get_dataloader("preprocessed", batch_size=8, action_chunk_size=16, obs_horizon=2)
    batch = next(iter(loader))
    print(f"  Batch: obs={batch['obs_history'].shape}, action={batch['action_chunk'].shape}")
    assert batch["obs_history"].shape == (8, 2, 1024)
    assert batch["action_chunk"].shape == (8, 16, 16)
    print("  ✓ DataLoader OK")

    # ── Test 3: Model Forward Pass ────────────────────────────────
    print("\n=== Test 3: Model Forward Pass ===")
    policy = DiffusionPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        action_chunk_size=16,
        obs_horizon=2,
        hidden_dim=256,
        time_emb_dim=128,
        num_blocks=3,
        dropout=0.1,
        num_timesteps=50,
        beta_schedule="cosine",
        device=device,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in policy.parameters()):,}")

    obs = batch["obs_history"].to(device)
    action = batch["action_chunk"].to(device)
    loss = policy(action, obs)
    print(f"  Loss: {loss.item():.4f}")
    assert loss.item() > 0
    print("  ✓ Forward pass OK")

    # ── Test 4: Sampling ─────────────────────────────────────────
    print("\n=== Test 4: Sampling ===")
    policy.eval()
    obs_single = sample["obs_history"].unsqueeze(0).to(device)

    action_ddpm = policy.sample_actions(obs_single, sampling_strategy="ddpm")
    print(f"  DDPM shape: {action_ddpm.shape}")
    assert action_ddpm.shape == (1, 16, 16)

    action_ddim = policy.sample_actions(obs_single, sampling_strategy="ddim", num_steps=10)
    print(f"  DDIM (10 steps) shape: {action_ddim.shape}")
    assert action_ddim.shape == (1, 16, 16)

    obs_batch = obs_single.repeat(4, 1, 1)
    action_batch = policy.sample_actions(obs_batch, sampling_strategy="ddim", num_steps=5)
    print(f"  Batch DDIM shape: {action_batch.shape}")
    assert action_batch.shape == (4, 16, 16)
    print("  ✓ Sampling OK")

    # ── Test 5: Checkpoint Save/Load ─────────────────────────────
    print("\n=== Test 5: Checkpoint Save/Load ===")
    ckpt = {
        "model_state": policy.state_dict(),
        "normalizer_min": normalizer.action_min,
        "normalizer_max": normalizer.action_max,
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "args": {
            "action_chunk_size": 16,
            "obs_horizon": 2,
            "hidden_dim": 256,
            "time_emb_dim": 128,
            "num_blocks": 3,
            "dropout": 0.1,
            "num_timesteps": 50,
            "beta_schedule": "cosine",
        },
    }
    tmp_path = os.path.join(tempfile.gettempdir(), "test_diffusion_policy.pt")
    torch.save(ckpt, tmp_path)
    print(f"  Saved: {tmp_path}")

    ckpt2 = torch.load(tmp_path, map_location=device, weights_only=False)
    assert ckpt2["obs_dim"] == 1024
    assert ckpt2["action_dim"] == 16
    os.remove(tmp_path)
    print("  ✓ Checkpoint OK")

    # ── Test 6: Training Loop (3 epochs) ────────────────────────
    print("\n=== Test 6: Training Loop (3 epochs × 20 batches) ===")
    small_loader, _ = get_dataloader("preprocessed", batch_size=16, action_chunk_size=16, obs_horizon=2)
    small_policy = DiffusionPolicy(
        obs_dim=1024, action_dim=16, action_chunk_size=16, obs_horizon=2,
        hidden_dim=128, time_emb_dim=64, num_blocks=2, dropout=0.1,
        num_timesteps=20, beta_schedule="cosine", device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(small_policy.parameters(), lr=1e-4)

    losses = []
    for epoch in range(3):
        small_policy.train()
        epoch_loss = 0.0
        count = 0
        for b in small_loader:
            if count >= 20:
                break
            obs_b = b["obs_history"].to(device)
            act_b = b["action_chunk"].to(device)
            optimizer.zero_grad()
            loss = small_policy(act_b, obs_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(small_policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
        avg = epoch_loss / count
        losses.append(avg)
        print(f"  Epoch {epoch+1}: loss={avg:.4f}")

    print(f"  Trend: {' '.join(f'{l:.4f}' for l in losses)}")
    print("  ✓ Training loop OK")

    # ── All done ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("Ready to run on server!")
    print("=" * 50)


if __name__ == "__main__":
    main()
