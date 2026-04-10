# Diffusion Policy — Dual-Arm Manipulation Imitation Learning

**Student 3: Diffusion Policy Core Model**

Implementation of a conditional Diffusion Policy for dual-arm robot manipulation, trained via imitation learning from expert demonstrations.

## Project Structure

```
.
├── diffusion_policy/
│   ├── models/
│   │   ├── policy.py      # DiffusionPolicy (main model)
│   │   ├── unet.py         # ConditionalUNet1D (noise predictor)
│   │   ├── diffusion.py    # DDPM/DDIM diffusion process
│   │   └── baselines.py    # BC and DAgger baselines
│   └── utils/
│       └── dataset.py      # RobotDataset, ActionNormalizer
├── scripts/
│   ├── train.py            # Training script
│   ├── inference.py        # Inference / sampling script
│   ├── eval.py             # Evaluation (MSE, MAE, smoothness)
│   ├── visualize.py        # Plotting (loss, trajectories, ablations)
│   └── run_ablation.py     # Ablation study runner
├── configs/
│   ├── train.yaml          # Main training config
│   └── ablation.yaml       # Ablation study configs
├── results/                # Checkpoints and outputs
├── AI_prompt_log.md         # Required AI tool usage log
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python scripts/train.py --data_dir preprocessed --num_epochs 100 --batch_size 32

# Run inference
python scripts/inference.py --checkpoint results/policy_final.pt --strategy ddpm

# Compare all policies (Diffusion vs BC vs DAgger)
python scripts/eval.py --compare_all --checkpoint results/policy_final.pt

# Ablation study
python scripts/run_ablation.py --study obs_horizon

# Visualizations
python scripts/visualize.py --loss results/train_log.txt
python scripts/visualize.py --trajectory --checkpoint results/policy_final.pt
python scripts/visualize.py --ablation results/ablation_obs_horizon_summary.json
```

## Architecture

### Conditional U-Net (Noise Predictor)
- **Input**: noisy action sequence `(B, T, 16)` + observation history `(B, obs_horizon, 1024)` + timestep `(B,)`
- **Conditioning**: Sinusoidal timestep embedding + observation embedding (FiLM-style addition)
- **Structure**: encoder (stride-2 convs) → residual middle → decoder (transposed convs) with skip connections
- **Output**: predicted noise `(B, T, 16)`

### Diffusion Process (DDPM)
- **Forward**: `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε`
- **Training**: MSE loss between predicted and true noise
- **Inference**: DDPM (100 steps) or DDIM (configurable, e.g. 10 steps)

### Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| `num_timesteps` | 100 |
| `beta_schedule` | cosine |
| `action_chunk_size` | 16 |
| `obs_horizon` | 2 |
| `hidden_dim` | 256 |
| `time_emb_dim` | 128 |
| `num_blocks` | 3 |

## Data Format

- `liftpot_actions.npy`: `(500, 75, 16)` — 500 demos × 75 timesteps × 16-dim actions
- `liftpot_images.npy`: `(500, 75, 1024)` — corresponding 1024-dim image features
- Actions normalized to `[-1, 1]` via min-max scaling from `stats.json`

## Experiments

1. **Action Chunk Size**: 8 vs 16 vs 32
2. **Observation Horizon**: 1 vs 2 vs 4 vs 8
3. **Number of Demonstrations**: 50 / 100 / 250 / 500
4. **Noise Schedule**: linear vs cosine
5. **Sampling Strategy**: DDPM vs DDIM (steps: 5/10/20/50)

## Results

See `results/` for trained checkpoints, loss curves, trajectory plots, and ablation summaries.

## Team Responsibilities

| Student | Contribution |
|---------|-------------|
| Student 1 | Environment setup, expert demonstrations, data preprocessing |
| Student 2 | Behavioral Cloning, DAgger, covariate shift analysis |
| **Student 3** | **Diffusion Policy model, training, inference, ablation** |
| Student 4 | Experiment pipeline, visualization, report |
