> 한국어 | **[English](README.en.md)**

# Hardware-specific Configurations

## Directory Structure

```
configs/
├── b200_8gpu/          # NVIDIA B200 × 8 (183GB each, ~1.47TB total)
├── h100_mig/           # NVIDIA H100 MIG 3g.40gb (single partition, ~42GB)
└── clickhouse-config.xml  # (unrelated to GPU training)
```

## b200_8gpu/

8× NVIDIA B200 환경용 설정. DDP/FSDP 멀티GPU 학습, FP8 지원.

| Config | Description |
|--------|-------------|
| `korean_3b_sft.yaml` | 3B SFT (Korean) |
| `korean_3b_fp8.yaml` | 3B FP8 학습 |
| `3b_pretrain.yaml` | 3B 사전학습 |
| `hybrid_3b.yaml` | 3B Hybrid (pretrain + SFT) |
| `korean_1b.yaml` | 1B Korean pretrain |
| `korean_1b_fp8.yaml` | 1B FP8 |
| `korean_1b_sft.yaml` | 1B SFT |
| `small.yaml` | Small model baseline |
| `small_fp8.yaml` | Small model FP8 |
| `medium.yaml` | Medium model |

Launch scripts: `scripts/b200_8gpu/`

## h100_mig/

H100 MIG 3g.40gb 단일 파티션 환경용 설정. BF16 + Gradient Checkpointing (FP8 미지원).

| Config | Description |
|--------|-------------|
| `korean_3b_sft_1gpu.yaml` | 3B SFT v2, single GPU, bs=4, grad_accum=7, NEFTune alpha=5.0 |
| `dpo_3b_1gpu.yaml` | 3B DPO Round 1, LoRA rank=32, beta=0.1, lr=5e-7, eff_batch=16 |

Launch scripts: `train_3b_sft_1gpu.sh`, `train_3b_dpo_1gpu.sh` (project root)

## SFT v2 학습 결과

`h100_mig/korean_3b_sft_1gpu.yaml`로 수행한 3B SFT v2 학습은 step 65,000에서 early stop 완료.

- **최종 val_loss**: 1.7924 (asymptote 도달)
- **체크포인트**: `checkpoints/3b_sft_v2/checkpoint-best`
- **중단 근거**: 50K 이후 개선폭이 측정 노이즈 수준(Δ < 0.001/5K steps)으로 감소

## DPO 설정

`h100_mig/dpo_3b_1gpu.yaml` — Nemotron-H 스타일 2-round DPO.

### Round 1 (yaml 기본값)

| 항목 | 값 | 근거 |
|------|-----|------|
| max_steps | 3,000 | 504K 샘플 중 48K 사용 (eff_batch=16) |
| batch_size | 1 | MIG VRAM 제약 |
| grad_accum_steps | 16 | eff_batch = 16 |
| lr | 5e-7 | DPO는 SFT보다 훨씬 낮은 lr 필요 |
| beta | 0.1 | DPO temperature (표준값) |
| max_length | 1024 | VRAM 제약으로 seq_len 제한 |
| lora_rank | 32 | |
| lora_alpha | 64 | scaling = alpha/rank = 2.0 |

### Round 2 (CLI 오버라이드)

```bash
--max_steps 2000 --beta 0.05 --lr 1e-7 --warmup_steps 50
```

더 보수적인 설정으로 정밀 조정. Round 1의 merged 체크포인트를 base로 사용.

### VRAM 예산 (실측)

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
# H100 MIG — SFT v2 완료, DPO 진행 중
bash train_3b_sft_1gpu.sh   # SFT (완료)
bash train_3b_dpo_1gpu.sh   # DPO Round 1 + 2 + SLERP Merge

# B200 8GPU (previous environment)
bash scripts/b200_8gpu/launch_3b_sft.sh
```

> 한국어 | **[English](README.en.md)**
