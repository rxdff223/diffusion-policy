"""
Training script for Diffusion Policy.
Usage:
    python scripts/train.py --config configs/train.yaml
"""
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from diffusion_policy.models.policy import DiffusionPolicy
from diffusion_policy.utils.dataset import get_dataloader


def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, normalizer = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        action_chunk_size=args.action_chunk_size,
        obs_horizon=args.obs_horizon,
        shuffle=True,
    )
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Model
    obs_dim = train_loader.dataset.obs_dim
    action_dim = train_loader.dataset.action_dim

    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_chunk_size=args.action_chunk_size,
        obs_horizon=args.obs_horizon,
        hidden_dim=args.hidden_dim,
        time_emb_dim=args.time_emb_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in policy.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.txt")

    for epoch in range(args.num_epochs):
        policy.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            obs = batch["obs_history"].to(device)      # (B, obs_horizon, obs_dim)
            action = batch["action_chunk"].to(device)  # (B, action_chunk_size, action_dim)

            optimizer.zero_grad()
            loss = policy.forward(action, obs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        log_line = f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"ckpt_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "policy_final.pt")
    torch.save({
        "model_state": policy.state_dict(),
        "normalizer_min": normalizer.action_min,
        "normalizer_max": normalizer.action_max,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "args": vars(args),
    }, final_path)
    print(f"Final model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="preprocessed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--action_chunk_size", type=int, default=16)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_timesteps", type=int, default=100)
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
