# AI Prompt Log — Diffusion Policy Implementation
**Student 3 | Dual-Arm Manipulation Imitation Learning Project**
**Date: 2026-04-10**

---

## Overview

All code for the Diffusion Policy core model was developed with assistance from Claude (Anthropic).
This log documents every significant prompt sent to the AI, the reasoning behind it, and the outcome.

---

## Prompt 1: Project Setup & Architecture Planning

**Prompt:**
> 我是第三个学生，负责实现 Diffusion Policy。我有一个 preprocessed 数据文件夹，包含 liftpot_actions.npy (500, 75, 16), liftpot_images.npy (500, 75, 1024)。请帮我创建一个完整的 Git 仓库，一步一步做，做完一步告诉我。

**Reasoning:**
- Needed to scaffold an entire research project from scratch
- Data format not immediately clear — needed to inspect shapes and stats.json
- Multiple students working on same repo — needed clear file structure

**Outcome:**
- Initialized git repo with clean directory structure
- Discovered data: 500 demos × 75 timesteps × 16-dim actions, 1024-dim image features
- Identified 16-dim action = dual-arm (7 joints × 2) + gripper states
- Created: `diffusion_policy/models/`, `diffusion_policy/utils/`, `scripts/`, `configs/`, `data/`, `results/`

---

## Prompt 2: Data Loader Implementation

**Prompt:**
> 实现数据加载器：RobotDataset，处理归一化、observation history window、action chunking。

**Reasoning:**
- Action chunking is essential for Diffusion Policy (predicting action sequences, not single steps)
- Need to handle the 16-dim action space normalization from [min, max] → [-1, 1]
- `valid_starts` calculation ensures we don't sample beyond trajectory end

**Outcome:**
- `diffusion_policy/utils/dataset.py`: RobotDataset, ActionNormalizer, get_dataloader
- Key design: each sample returns (obs_history[obs_horizon, obs_dim], action_chunk[action_chunk_size, action_dim])
- `action_chunk_size=16, obs_horizon=2` as defaults matching the paper

---

## Prompt 3: U-Net Backbone

**Prompt:**
> 实现 Conditional U-Net backbone，给噪声预测网络用。输入：noisy action sequence + observation conditioning + timestep embedding。输出：predicted noise。

**Reasoning:**
- Following the Diffusion Policy paper architecture: 1D U-Net with temporal convolution
- Observation conditioning via FiLM (Feature-wise Linear Modulation): add time embedding with observation embedding
- Need sinusoidal positional embedding for diffusion timesteps
- Residual blocks with GroupNorm (not BatchNorm) — works better for small batch sizes

**Outcome:**
- `diffusion_policy/models/unet.py`: ConditionalUNet1D with encoder-decoder structure
- SinusoidalPosEmb for timestep encoding
- ResidualBlock with GroupNorm + SiLU + time conditioning
- Critical bug found and fixed: GroupNorm was applied to wrong channel dimension (in_channels vs out_channels)

---

## Prompt 4: Diffusion Process (DDPM)

**Prompt:**
> 实现 DDPM diffusion process：噪声调度（cosine/linear）、前向扩散、反向去噪、训练损失、推理采样循环。

**Reasoning:**
- Following DDPM (Ho et al., 2020) formulation
- Cosine schedule recommended for better stability than linear (as in paper)
- Need both DDPM and DDIM sampling strategies (DDIM = faster inference)
- Key challenge: broadcasting shape mismatches when combining (B,) tensors with (B, T, action_dim)

**Outcome:**
- `diffusion_policy/models/diffusion.py`: DiffusionProcess class
- get_noise_schedule() with linear and cosine options
- compute_loss(): predict noise given noisy action + conditioning
- sample(): full DDPM and DDIM inference loops
- Bug fixed: alphas/betas/beta need `[:, None, None]` reshape to broadcast with action tensors

---

## Prompt 5: Full Model + Training Script

**Prompt:**
> 把 U-Net + Diffusion 组合成 DiffusionPolicy 主模型，然后写训练脚本。

**Reasoning:**
- Policy class wraps noise_predictor + diffusion process
- Training: AdamW optimizer + CosineAnnealingLR (standard for diffusion models)
- Gradient clipping at 1.0 for stability
- Save checkpoints every N epochs + final model with normalizer

**Outcome:**
- `diffusion_policy/models/policy.py`: DiffusionPolicy
- `scripts/train.py`: full training loop with tqdm progress bar, logging
- `scripts/inference.py`: loading checkpoint + running DDPM/DDIM sampling

---

## Prompt 6: Architecture Debug — Skip Connection Ordering

**Prompt:**
> 运行 sanity check 时出现 RuntimeError: Sizes of tensors must match... Expected size 8 but got size 4 in skip connection cat。

**Reasoning:**
- U-Net encoder halves sequence length at each downsample step
- Decoder upsamples back — skip connections must have matching sequence lengths
- Was storing skip AFTER downsample (shorter length) → after upsample still didn't match
- Fix: store skip BEFORE downsample, so lengths match after upsample

**Outcome:**
- Changed `skips.append(x)` to happen before `x = conv(x)` in encoder loop
- Verified: T=16 → downsample→ T=8 → upsample→ T=16 (matches stored skip T=16)

---

## Prompt 7: Broadcasting Bug in reverse_step

**Prompt:**
> 运行 inference 时出现 RuntimeError: The size of tensor a (4) must match the size of tensor b (16) at non-singleton dimension 2。

**Reasoning:**
- `self.alphas_cumprod[timesteps]` returns shape (B,) — scalar per sample
- Multiplying with `noise_pred` of shape (B, T, action_dim) caused broadcasting failure
- Solution: reshape all scalar schedule values with `[:, None, None]`

**Outcome:**
- Fixed `reverse_step()`: all schedule values now `(B, 1, 1)` for correct broadcasting
- Fixed `p_mean_variance()`: same pattern

---

## Prompt 8: Ablation Study Scripts

**Prompt:**
> 创建消融实验脚本，遍历不同的 action_chunk_size、obs_horizon、demonstration数量、noise schedule、sampling strategy。

**Reasoning:**
- Ablation studies are required by the project spec
- Need to sweep 5 dimensions systematically
- Each variant should save its own checkpoint and config

**Outcome:**
- `configs/ablation.yaml`: structured config for all 5 ablation studies
- `scripts/run_ablation.py`: generates variants, trains each, saves results JSON
- Studies: action_chunk (8/16/32), obs_horizon (1/2/4/8), num_demos (50/100/250/500), noise_schedule, sampling

---

## Prompt 9: Visualization Script

**Prompt:**
> 创建可视化脚本：训练损失曲线、预测 vs 真实轨迹对比图、消融实验柱状图、action heatmap。

**Reasoning:**
- Visualizations needed for the final report
- Trajectory comparison shows how well policy captures temporal dynamics
- Ablation bar charts needed for experiment section
- Heatmaps show per-dimension prediction quality

**Outcome:**
- `scripts/visualize.py`: 4 plotting functions using matplotlib
- Supports `--loss`, `--trajectory`, `--ablation`, `--heatmap` flags
- Saves to `results/trajectory_plots/` and `results/heatmap_plots/`

---

## Prompt 10: Evaluation Script

**Prompt:**
> 创建评估脚本：计算 MSE/MAE、轨迹平滑度（jerk）、与 BC/DAgger baseline 的对比。

**Reasoning:**
- Evaluation is required for comparison section
- Smoothness metric (jerk = 3rd derivative) measures action quality
- Need to support both single-policy eval and multi-policy comparison

**Outcome:**
- `scripts/eval.py`: compute_trajectory_smoothness(), evaluate_policy(), compare_all_policies()
- Outputs comparison table and JSON summary
- Supports BC, DAgger, Diffusion (DDPM), Diffusion (DDIM) comparison

---

## Prompt 11: Behavioral Cloning & DAgger Baselines

**Prompt:**
> 实现 BC 和 DAgger baseline 模型（用于与 Diffusion Policy 对比）。

**Reasoning:**
- BC/DAgger are the baseline methods for comparison (as specified in project)
- Student 2 is responsible for their implementation, but Student 3 needs them for evaluation
- Simple MLP architecture is sufficient for baseline comparison

**Outcome:**
- `diffusion_policy/models/baselines.py`: BCPolicy, DAggerPolicy
- Simple MLP with LayerNorm and dropout
- Forward pass: flatten obs_history → MLP → action
- Same interface as DiffusionPolicy for easy comparison

---

## Prompt 12: Complete Project — "全部做完"

**Prompt:**
> 全部做完吧

**Reasoning:**
- Need to complete all remaining components: ablation configs, baselines, eval, visualization, AI prompt log

**Outcome:**
- All components implemented and committed
- Git repo in clean state with descriptive commits

---

## Model Architecture Summary

```
DiffusionPolicy
├── ConditionalUNet1D
│   ├── SinusoidalPosEmb (timestep → time_emb_dim)
│   ├── obs_encoder (Linear × 2, mean pool)
│   ├── action_in (Conv1d action_dim → hidden_dim)
│   ├── encoder: [Conv1d stride=2 → ResidualBlock] × num_blocks
│   ├── middle: ResidualBlock
│   ├── decoder: [ConvTranspose1d → concat(skip) → ResidualBlock] × num_blocks
│   └── out: GroupNorm → SiLU → Conv1d hidden_dim → action_dim
└── DiffusionProcess
    ├── get_noise_schedule (cosine/linear)
    ├── forward_diffusion: x_t = sqrt(ᾱ_t)x₀ + sqrt(1-ᾱ_t)ε
    ├── compute_loss: MSE(noise_pred, ε)
    └── sample: DDPM (100 steps) / DDIM (configurable steps)
```

## Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_timesteps | 100 | Balance between quality and speed |
| beta_schedule | cosine | More gradual noise addition (Nichol & Dhariwal, 2021) |
| action_chunk_size | 16 | Predict 16 future actions per step |
| obs_horizon | 2 | 2-frame observation history |
| hidden_dim | 256 | Standard for diffusion policies |
| time_emb_dim | 128 | Double sinusoidal embedding |
| lr | 1e-4 | Standard for diffusion models |
| weight_decay | 1e-4 | Regularization |
