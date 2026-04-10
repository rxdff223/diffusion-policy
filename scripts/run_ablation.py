"""
Ablation study runner.
Sweeps over hyperparameter configurations and saves results.
Usage:
    python scripts/run_ablation.py --study action_chunk
    python scripts/run_ablation.py --study obs_horizon
    python scripts/run_ablation.py --study num_demos
"""
import argparse
import copy
import json
import os
import sys
import yaml
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_policy.models.policy import DiffusionPolicy
from diffusion_policy.utils.dataset import get_dataloader, RobotDataset


def build_variants(base_cfg, study_name):
    """Generate config variants for a given ablation study."""
    variants = []
    if study_name == "action_chunk":
        for chunk_size in base_cfg["action_chunk_size"]:
            cfg = copy.deepcopy(base_cfg)
            cfg["action_chunk_size"] = chunk_size
            cfg["experiment_name"] = f"action_chunk_{chunk_size}"
            variants.append(cfg)
    elif study_name == "obs_horizon":
        for obs_h in base_cfg["obs_horizon"]:
            cfg = copy.deepcopy(base_cfg)
            cfg["obs_horizon"] = obs_h
            cfg["experiment_name"] = f"obs_horizon_{obs_h}"
            variants.append(cfg)
    elif study_name == "num_demos":
        for n in base_cfg["num_demos"]:
            cfg = copy.deepcopy(base_cfg)
            cfg["num_demos"] = n
            cfg["experiment_name"] = f"num_demos_{n}"
            variants.append(cfg)
    elif study_name == "noise_schedule":
        for schedule in base_cfg["beta_schedule"]:
            for T in base_cfg["num_timesteps"]:
                cfg = copy.deepcopy(base_cfg)
                cfg["beta_schedule"] = schedule
                cfg["num_timesteps"] = T
                cfg["experiment_name"] = f"schedule_{schedule}_T{T}"
                variants.append(cfg)
    elif study_name == "sampling":
        for strategy in base_cfg["sampling_strategy"]:
            for steps in base_cfg["ddim_num_steps"]:
                cfg = copy.deepcopy(base_cfg)
                cfg["sampling_strategy"] = strategy
                cfg["ddim_num_steps"] = steps
                cfg["experiment_name"] = f"sampling_{strategy}_steps{steps}"
                variants.append(cfg)
    return variants


def train_one(cfg, device="cuda"):
    """Train a single variant and return final loss."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: {cfg['experiment_name']}")
    print(f"{'='*60}")

    train_loader, normalizer = get_dataloader(
        data_dir=cfg.get("data_dir", "preprocessed"),
        batch_size=cfg.get("batch_size", 32),
        action_chunk_size=cfg.get("action_chunk_size", 16),
        obs_horizon=cfg.get("obs_horizon", 2),
        shuffle=True,
    )

    obs_dim = train_loader.dataset.obs_dim
    action_dim = train_loader.dataset.action_dim

    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_chunk_size=cfg.get("action_chunk_size", 16),
        obs_horizon=cfg.get("obs_horizon", 2),
        hidden_dim=cfg.get("hidden_dim", 256),
        time_emb_dim=cfg.get("time_emb_dim", 128),
        num_blocks=cfg.get("num_blocks", 3),
        dropout=cfg.get("dropout", 0.1),
        num_timesteps=cfg.get("num_timesteps", 100),
        beta_schedule=cfg.get("beta_schedule", "cosine"),
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("num_epochs", 100), eta_min=1e-6
    )

    results_dir = os.path.join(cfg.get("output_dir", "results"), cfg["experiment_name"])
    os.makedirs(results_dir, exist_ok=True)

    loss_history = []
    for epoch in range(cfg.get("num_epochs", 100)):
        policy.train()
        epoch_loss = 0.0
        for batch in train_loader:
            obs = batch["obs_history"].to(device)
            action = batch["action_chunk"].to(device)
            optimizer.zero_grad()
            loss = policy.forward(action, obs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

    # Save model
    torch.save({
        "model_state": policy.state_dict(),
        "normalizer_min": normalizer.action_min,
        "normalizer_max": normalizer.action_max,
        "config": cfg,
        "loss_history": loss_history,
    }, os.path.join(results_dir, "policy_final.pt"))

    print(f"  Final loss: {loss_history[-1]:.6f}")
    return loss_history


def run_ablation(study_name, data_dir="preprocessed", output_dir="results"):
    with open("configs/ablation.yaml") as f:
        all_configs = yaml.safe_load(f)

    base_cfg = all_configs[f"ablation_{study_name}"]
    base_cfg["data_dir"] = data_dir
    base_cfg["output_dir"] = output_dir

    variants = build_variants(base_cfg, study_name)
    print(f"Running {len(variants)} variants for study '{study_name}'")

    results = {}
    for cfg in variants:
        try:
            loss_hist = train_one(cfg)
            results[cfg["experiment_name"]] = {
                "final_loss": float(loss_hist[-1]),
                "min_loss": float(min(loss_hist)),
                "config": {k: v for k, v in cfg.items() if k not in ["data_dir", "output_dir"]},
            }
        except Exception as e:
            print(f"  ERROR in {cfg['experiment_name']}: {e}")
            results[cfg["experiment_name"]] = {"error": str(e)}

    # Save results summary
    summary_path = os.path.join(output_dir, f"ablation_{study_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=str, required=True,
                        choices=["action_chunk", "obs_horizon", "num_demos",
                                 "noise_schedule", "sampling"])
    parser.add_argument("--data_dir", type=str, default="preprocessed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_ablation(args.study, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
