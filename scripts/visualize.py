"""
Visualization script for Diffusion Policy experiments.
Generates:
  - Training loss curves
  - Predicted vs ground-truth action trajectory comparison
  - Ablation study bar charts
  - Sample action heatmaps

Usage:
    python scripts/visualize.py --loss results/train_log.txt
    python scripts/visualize.py --trajectory --checkpoint results/policy_final.pt
    python scripts/visualize.py --ablation results/ablation_obs_horizon_summary.json
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_policy.models.policy import DiffusionPolicy
from diffusion_policy.utils.dataset import RobotDataset


def plot_loss_curve(log_path, save_path=None):
    """Parse train_log.txt and plot loss over epochs."""
    epochs, losses = [], []
    with open(log_path) as f:
        for line in f:
            if "Epoch" in line and "Loss" in line:
                parts = line.strip().split("|")
                ep = int(parts[0].split()[1])
                loss = float(parts[1].split(":")[1].strip())
                epochs.append(ep)
                losses.append(loss)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title("Diffusion Policy Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    path = save_path or log_path.replace(".txt", "_loss.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_trajectory_comparison(checkpoint_path, data_dir, num_samples=5,
                                save_dir="results/trajectory_plots"):
    """Compare predicted vs ground-truth action trajectories."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    normalizer_min = torch.tensor(ckpt["normalizer_min"], dtype=torch.float32)
    normalizer_max = torch.tensor(ckpt["normalizer_max"], dtype=torch.float32)

    policy = DiffusionPolicy(
        obs_dim=args.get("obs_dim", 1024),
        action_dim=args.get("action_dim", 16),
        action_chunk_size=args.get("action_chunk_size", 16),
        obs_horizon=args.get("obs_horizon", 2),
        hidden_dim=args.get("hidden_dim", 256),
        time_emb_dim=args.get("time_emb_dim", 128),
        num_blocks=args.get("num_blocks", 3),
        dropout=args.get("dropout", 0.1),
        num_timesteps=args.get("num_timesteps", 100),
        beta_schedule=args.get("beta_schedule", "cosine"),
        device=device,
    ).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    # Load dataset
    dataset = RobotDataset(data_dir)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    for idx in indices:
        sample = dataset[idx]
        obs = sample["obs_history"].unsqueeze(0).to(device)
        gt_action = sample["action_chunk"].numpy()

        # Predict
        with torch.no_grad():
            pred_norm = policy.sample_actions(obs, sampling_strategy="ddpm")[0].cpu().numpy()

        # Denormalize
        action_range = normalizer_max - normalizer_min
        gt_action_denorm = (gt_action + 1) / 2 * action_range.numpy() + normalizer_min.numpy()
        pred_denorm = (pred_norm + 1) / 2 * action_range.numpy() + normalizer_min.numpy()

        # Plot first 4 action dimensions
        n_dims = min(4, gt_action_denorm.shape[1])
        fig, axes = plt.subplots(n_dims, 1, figsize=(10, n_dims * 1.5), sharex=True)
        if n_dims == 1:
            axes = [axes]

        for d in range(n_dims):
            t = np.arange(gt_action_denorm.shape[0])
            axes[d].plot(t, gt_action_denorm[:, d], "b-", label="Ground Truth", linewidth=1.5)
            axes[d].plot(t, pred_denorm[:, d], "r--", label="Predicted", linewidth=1.5, alpha=0.8)
            axes[d].set_ylabel(f"Action {d}")
            axes[d].legend(fontsize=8)
            axes[d].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timestep (within chunk)")
        fig.suptitle(f"Trajectory Comparison — Sample {idx}")
        fig.tight_layout()
        path = os.path.join(save_dir, f"trajectory_{idx}.png")
        fig.savefig(path, dpi=120)
        plt.close()
        print(f"Saved: {path}")


def plot_ablation_results(summary_path, save_path=None):
    """Plot ablation study results as grouped bar charts."""
    with open(summary_path) as f:
        data = json.load(f)

    # Extract experiment names and final losses
    names, losses = [], []
    for name, res in data.items():
        if "error" in res:
            continue
        # Shorten name for display
        short = name.replace("ablation_", "").replace("_", "\n")
        names.append(short)
        losses.append(res["final_loss"])

    names = names[:8]  # limit to 8 bars per chart
    losses = losses[:8]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars = ax.bar(names, losses, color=colors, edgecolor="navy", linewidth=0.5)

    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{loss:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Final Training Loss")
    ax.set_title(os.path.basename(summary_path).replace("_summary.json", "").replace("_", " ").title())
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    path = save_path or summary_path.replace(".json", ".png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_action_heatmap(checkpoint_path, data_dir, num_samples=3,
                         save_dir="results/heatmap_plots"):
    """Plot heatmaps of predicted vs ground-truth action chunks."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    normalizer_min = torch.tensor(ckpt["normalizer_min"], dtype=torch.float32)
    normalizer_max = torch.tensor(ckpt["normalizer_max"], dtype=torch.float32)

    policy = DiffusionPolicy(
        obs_dim=args.get("obs_dim", 1024),
        action_dim=args.get("action_dim", 16),
        action_chunk_size=args.get("action_chunk_size", 16),
        obs_horizon=args.get("obs_horizon", 2),
        hidden_dim=args.get("hidden_dim", 256),
        time_emb_dim=args.get("time_emb_dim", 128),
        num_blocks=args.get("num_blocks", 3),
        dropout=args.get("dropout", 0.1),
        num_timesteps=args.get("num_timesteps", 100),
        beta_schedule=args.get("beta_schedule", "cosine"),
        device=device,
    ).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    dataset = RobotDataset(data_dir)
    rng = np.random.default_rng(123)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(14, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        obs = sample["obs_history"].unsqueeze(0).to(device)
        gt = sample["action_chunk"].numpy()

        with torch.no_grad():
            pred = policy.sample_actions(obs, sampling_strategy="ddpm")[0].cpu().numpy()

        # Denormalize
        rng_n = normalizer_max - normalizer_min
        gt_d = (gt + 1) / 2 * rng_n.numpy() + normalizer_min.numpy()
        pred_d = (pred + 1) / 2 * rng_n.numpy() + normalizer_min.numpy()
        error = np.abs(gt_d - pred_d)

        for ax, mat, title in zip(
            axes[row], [gt_d, pred_d, error],
            ["Ground Truth", "Predicted", "Absolute Error"]
        ):
            im = ax.imshow(mat.T, aspect="auto", cmap="viridis" if "Error" not in title else "Reds")
            ax.set_title(title, fontsize=9)
            ax.set_ylabel("Action Dim")
            ax.set_xlabel("Timestep")
            plt.colorbar(im, ax=ax, shrink=0.8)

        axes[row, 0].set_ylabel(f"Sample {idx}\nAction Dim")

    fig.suptitle("Action Chunk Comparison (denormalized)")
    fig.tight_layout()
    path = os.path.join(save_dir, "action_heatmaps.png")
    fig.savefig(path, dpi=120)
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, help="Path to train_log.txt")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--ablation", type=str, help="Path to ablation summary .json")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="results/policy_final.pt")
    parser.add_argument("--data_dir", type=str, default="preprocessed")
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    if args.loss:
        plot_loss_curve(args.loss)

    if args.trajectory:
        plot_trajectory_comparison(args.checkpoint, args.data_dir, save_dir=args.save_dir)

    if args.ablation:
        plot_ablation_results(args.ablation)

    if args.heatmap:
        plot_action_heatmap(args.checkpoint, args.data_dir, save_dir=args.save_dir)

    if not (args.loss or args.trajectory or args.ablation or args.heatmap):
        parser.print_help()


if __name__ == "__main__":
    main()
