> **[한국어](README.md)** | English

# Hardware-specific Configurations

## Directory Structure

```
configs/
├── b200_8gpu/          # NVIDIA B200 × 8 (183GB each, ~1.47TB total)
├── h100_mig/           # NVIDIA H100 MIG 3g.40gb (single partition, ~42GB)
└── clickhouse-config.xml  # (unrelated to GPU training)
```

## b200_8gpu/

Configurations for 8× NVIDIA B200 environment. Multi-GPU training with DDP/FSDP, FP8 support.

| Config | Description |
|--------|-------------|
| `korean_3b_sft.yaml` | 3B SFT (Korean) |
| `korean_3b_fp8.yaml` | 3B FP8 training |
| `3b_pretrain.yaml` | 3B pretraining |
| `hybrid_3b.yaml` | 3B Hybrid (pretrain + SFT) |
| `korean_1b.yaml` | 1B Korean pretrain |
| `korean_1b_fp8.yaml` | 1B FP8 |
| `korean_1b_sft.yaml` | 1B SFT |
| `small.yaml` | Small model baseline |
| `small_fp8.yaml` | Small model FP8 |
| `medium.yaml` | Medium model |

Launch scripts: `scripts/b200_8gpu/`

## h100_mig/

Configurations for H100 MIG 3g.40gb single partition environment. BF16 + Gradient Checkpointing (FP8 not supported).

| Config | Description |
|--------|-------------|
| `korean_3b_sft_1gpu.yaml` | 3B SFT v2, single GPU, bs=4, grad_accum=7, NEFTune alpha=5.0 |
| `dpo_3b_1gpu.yaml` | 3B DPO Round 1, LoRA rank=32, beta=0.1, lr=5e-7, eff_batch=16 |

Launch scripts: `train_3b_sft_1gpu.sh`, `train_3b_dpo_1gpu.sh` (project root)

## SFT v2 Training Results

The 3B SFT v2 training conducted with `h100_mig/korean_3b_sft_1gpu.yaml` completed early stopping at step 65,000.

- **Final val_loss**: 1.7924 (asymptote reached)
- **Checkpoint**: `checkpoints/3b_sft_v2/checkpoint-best`
- **Stopping rationale**: Improvement after 50K declined to measurement noise level (Δ < 0.001/5K steps)

## DPO Configuration

`h100_mig/dpo_3b_1gpu.yaml` — Nemotron-H style 2-round DPO.

### Round 1 (YAML defaults)

| Item | Value | Rationale |
|------|-------|-----------|
| max_steps | 3,000 | Uses 48K of 504K samples (eff_batch=16) |
| batch_size | 1 | MIG VRAM constraint |
| grad_accum_steps | 16 | eff_batch = 16 |
| lr | 5e-7 | DPO requires much lower lr than SFT |
| beta | 0.1 | DPO temperature (standard) |
| max_length | 1024 | VRAM constraint limits seq_len |
| lora_rank | 32 | |
| lora_alpha | 64 | scaling = alpha/rank = 2.0 |

### Round 2 (CLI override)

```bash
--max_steps 2000 --beta 0.05 --lr 1e-7 --warmup_steps 50
```

Fine-tuning with more conservative settings. Uses merged checkpoint from Round 1 as base.
β lowered from 0.1→0.05 to reduce deviation from reference model, preventing over-alignment.
lr reduced 10× (5e-7→1e-7) to preserve SFT knowledge while fine-tuning preferences.

### VRAM Budget (measured)

```
Base model (bf16):     ~6.0 GB
LoRA adapters:         ~0.08 GB
Optimizer (AdamW):     ~0.16 GB
Activations + grad:    ~0.1 GB
───────────────────────────────
Total:                 ~6.3 GB / 42.3 GB (15%)
```

## Usage

```bash
# H100 MIG — SFT v2 complete, DPO R1+R2 complete, SLERP merged
bash train_3b_sft_1gpu.sh   # SFT (complete)
bash train_3b_dpo_1gpu.sh   # DPO Round 1 + 2 + SLERP Merge (complete)
# Final model: checkpoints/3b_dpo/checkpoint-slerp

# B200 8GPU (previous environment)
bash scripts/b200_8gpu/launch_3b_sft.sh
```

> **[한국어](README.md)** | English
