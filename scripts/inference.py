"""
Inference script for Diffusion Policy.
Usage:
    python scripts/inference.py --checkpoint results/policy_final.pt --strategy ddpm
"""
import argparse
import torch
import numpy as np

from diffusion_policy.models.policy import DiffusionPolicy
from diffusion_policy.utils.dataset import RobotDataset


def load_policy(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = ckpt.get("args", {})
    normalizer_min = ckpt["normalizer_min"]
    normalizer_max = ckpt["normalizer_max"]

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

    from diffusion_policy.utils.dataset import ActionNormalizer
    normalizer = ActionNormalizer(
        torch.tensor(normalizer_min, dtype=torch.float32),
        torch.tensor(normalizer_max, dtype=torch.float32),
    )

    # Return dataset params so caller can use them
    dataset_kwargs = {
        "action_chunk_size": args.get("action_chunk_size", 16),
        "obs_horizon": args.get("obs_horizon", 2),
    }
    return policy, normalizer, dataset_kwargs


@torch.no_grad()
def inference_demo(policy, normalizer, data_dir, dataset_kwargs,
                  num_samples=5, sampling_strategy="ddpm", num_steps=None):
    """
    Run inference on random samples from the dataset and print results.
    """
    dataset = RobotDataset(data_dir, **dataset_kwargs)

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]

        obs = sample["obs_history"].unsqueeze(0).to(policy.device)
        gt_action_norm = sample["action_chunk"]

        action_norm = policy.sample_actions(obs, sampling_strategy, num_steps)
        action_norm = action_norm[0].cpu().numpy()

        gt_action = normalizer.denormalize(gt_action_norm.numpy())
        pred_action = normalizer.denormalize(action_norm)

        print(f"\n--- Sample {i+1} (idx={idx}) ---")
        print(f"GT action (first 3 dims, first 3 steps):\n{gt_action[:3, :3]}")
        print(f"Pred action (first 3 dims, first 3 steps):\n{pred_action[:3, :3]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="preprocessed")
    parser.add_argument("--strategy", type=str, default="ddpm",
                        choices=["ddpm", "ddim"])
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Number of steps for DDIM sampling (default: full)")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sampling strategy: {args.strategy}")
    if args.strategy == "ddim":
        print(f"DDIM steps: {args.num_steps}")

    policy, normalizer, dataset_kwargs = load_policy(args.checkpoint, device)
    print("Policy loaded successfully.")

    inference_demo(policy, normalizer, args.data_dir, dataset_kwargs,
                   num_samples=args.num_samples,
                   sampling_strategy=args.strategy,
                   num_steps=args.num_steps)


if __name__ == "__main__":
    main()
