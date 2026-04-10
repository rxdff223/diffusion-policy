"""
Evaluation script for Diffusion Policy.
Computes:
  - MSE / MAE per action dimension
  - Trajectory smoothness (velocity/acceleration Jerk)
  - Comparison with BC and DAgger baselines
  - Per-demonstration success metrics

Usage:
    python scripts/eval.py --checkpoint results/policy_final.pt --data_dir preprocessed
    python scripts/eval.py --compare_all --data_dir preprocessed
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_policy.models.policy import DiffusionPolicy
from diffusion_policy.models.baselines import BCPolicy, DAggerPolicy
from diffusion_policy.utils.dataset import RobotDataset


def load_diffusion_policy(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)
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

    def denormalizer(x):
        rng = normalizer_max - normalizer_min
        return (x + 1) / 2 * rng + normalizer_min

    return policy, denormalizer


def load_bc_policy(obs_dim, action_dim, obs_horizon, checkpoint_path=None, device="cuda"):
    policy = BCPolicy(obs_dim, action_dim, hidden_dim=256, obs_horizon=obs_horizon).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(ckpt["model_state"])
    policy.eval()
    return policy


def compute_trajectory_smoothness(actions):
    """Compute jerk (3rd derivative) as smoothness metric. Lower = smoother."""
    if actions.shape[0] < 4:
        return np.nan
    vel = np.diff(actions, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    return float(np.mean(np.square(jerk)))


def evaluate_policy(policy, dataset, device, denormalizer=None,
                    sampling_strategy="ddpm", num_steps=None,
                    policy_type="diffusion"):
    """
    Evaluate a policy on all dataset samples.
    Returns dict of metrics.
    """
    results = {
        "mse_per_dim": [],
        "mae_per_dim": [],
        "total_mse": [],
        "total_mae": [],
        "smoothness": [],
        "chunks_evaluated": 0,
    }

    for idx in range(len(dataset)):
        sample = dataset[idx]
        obs = sample["obs_history"].unsqueeze(0).to(device)
        gt_action = sample["action_chunk"]  # normalized [-1, 1]

        if denormalizer is not None:
            gt_denorm = denormalizer(gt_action).cpu().numpy()
        else:
            gt_denorm = gt_action.numpy()

        # Predict
        with torch.no_grad():
            if policy_type == "diffusion":
                pred_norm = policy.sample_actions(obs, sampling_strategy, num_steps)[0].cpu().numpy()
            else:
                # BC / DAgger: predict single action, tile to match chunk
                pred_single = policy(obs).cpu()
                pred_norm = pred_single.repeat(dataset.action_chunk_size, 1).T.numpy()

        if denormalizer is not None:
            pred_denorm = denormalizer(torch.tensor(pred_norm)).cpu().numpy()
        else:
            pred_denorm = pred_norm

        # Metrics
        mse = np.mean((gt_denorm - pred_denorm) ** 2, axis=0)  # per dim
        mae = np.mean(np.abs(gt_denorm - pred_denorm), axis=0)
        smooth = compute_trajectory_smoothness(pred_denorm)

        results["mse_per_dim"].append(mse)
        results["mae_per_dim"].append(mae)
        results["total_mse"].append(float(np.mean(mse)))
        results["total_mae"].append(float(np.mean(mae)))
        results["smoothness"].append(smooth)
        results["chunks_evaluated"] += 1

    # Aggregate
    results["mean_mse"] = float(np.mean(results["total_mse"]))
    results["mean_mae"] = float(np.mean(results["total_mae"]))
    results["mean_smoothness"] = float(np.nanmean(results["smoothness"]))
    results["mse_std"] = float(np.std(results["total_mse"]))
    results["mae_std"] = float(np.std(results["total_mae"]))

    # Per-dimension averages
    results["avg_mse_per_dim"] = [float(np.mean([m[i] for m in results["mse_per_dim"]]))
                                   for i in range(dataset.action_dim)]

    return results


def compare_all_policies(data_dir, diffusion_ckpt, output_dir="results", device="cuda"):
    """Compare Diffusion Policy vs BC vs DAgger baselines."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    dataset = RobotDataset(data_dir)

    all_results = {}

    # --- Diffusion Policy ---
    print("\nEvaluating Diffusion Policy...")
    policy, denorm = load_diffusion_policy(diffusion_ckpt, device)
    diff_results = evaluate_policy(policy, dataset, device, denorm,
                                   sampling_strategy="ddpm")
    all_results["diffusion_ddpm"] = diff_results
    print(f"  Diffusion (DDPM): MSE={diff_results['mean_mse']:.6f}, "
          f"MAE={diff_results['mean_mae']:.6f}, Smoothness={diff_results['mean_smoothness']:.6f}")

    # --- Diffusion Policy DDIM ---
    print("Evaluating Diffusion Policy (DDIM, 10 steps)...")
    ddim_results = evaluate_policy(policy, dataset, device, denorm,
                                   sampling_strategy="ddim", num_steps=10)
    all_results["diffusion_ddim10"] = ddim_results
    print(f"  Diffusion (DDIM×10): MSE={ddim_results['mean_mse']:.6f}, "
          f"MAE={ddim_results['mean_mae']:.6f}")

    # --- BC baseline ---
    print("Evaluating BC baseline...")
    bc_policy = load_bc_policy(dataset.obs_dim, dataset.action_dim, dataset.obs_horizon,
                               device=device)
    bc_results = evaluate_policy(bc_policy, dataset, device, denorm=None,
                                 policy_type="bc")
    all_results["bc"] = bc_results
    print(f"  BC: MSE={bc_results['mean_mse']:.6f}, "
          f"MAE={bc_results['mean_mae']:.6f}, Smoothness={bc_results['mean_smoothness']:.6f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "evaluation_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'Policy':<22} {'MSE':>10} {'MAE':>10} {'Smoothness':>12}")
    print("-" * 65)
    for name, res in all_results.items():
        print(f"{name:<22} {res['mean_mse']:>10.6f} {res['mean_mae']:>10.6f} "
              f"{res['mean_smoothness']:>12.6f}")
    print("=" * 65)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/policy_final.pt")
    parser.add_argument("--data_dir", type=str, default="preprocessed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--compare_all", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.compare_all:
        compare_all_policies(args.data_dir, args.checkpoint, args.output_dir, device)
    else:
        # Single policy evaluation
        dataset = RobotDataset(args.data_dir)
        policy, denorm = load_diffusion_policy(args.checkpoint, device)
        results = evaluate_policy(policy, dataset, device, denorm)
        print(f"\nResults for {args.checkpoint}:")
        print(f"  Mean MSE: {results['mean_mse']:.6f} ± {results['mse_std']:.6f}")
        print(f"  Mean MAE: {results['mean_mae']:.6f} ± {results['mae_std']:.6f}")
        print(f"  Mean Smoothness (jerk): {results['mean_smoothness']:.6f}")
        print(f"  Chunks evaluated: {results['chunks_evaluated']}")


if __name__ == "__main__":
    main()
