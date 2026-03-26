> **[한국어](README.md)** | English

<div align="center">

# EVAFRILL-Mo

**Hybrid Mamba-2 + Transformer Language Model**

*Bride Eva (Bride of Frankenstein) + FRIDAY (Iron Man's AI assistant) + LLM + Nemotron's Mo*

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10.0--nv25.12-EE4C2C?logo=pytorch&logoColor=white)
![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![FlashAttention 2](https://img.shields.io/badge/FlashAttention-2.7.4-blueviolet)
![FP8](https://img.shields.io/badge/FP8-Native-orange)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![GPUs](https://img.shields.io/badge/GPUs-7%C3%97%20B200-76B900?logo=nvidia&logoColor=white)
![Model](https://img.shields.io/badge/Model-2.94B%20params-blue)
![Training](https://img.shields.io/badge/Training-319K%20steps-brightgreen)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-pathcosmos%2FEVAFRILL--Mo--3B-yellow)](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

**Model download**: [🤗 HuggingFace Hub](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

The HF Hub contains **7 model versions** + LoRA weights + preference data + training configs/scripts for full reproducibility:

| Directory | Model | Description |
|-----------|-------|-------------|
| `slerp/` | ⭐ **Recommended** | SFT + DPO SLERP merge (α=0.5) |
| `pretrain/` | Pretrain | 319K steps, 55B tokens |
| `sft-v2/` | SFT v2 | 65K steps, val_loss 1.79 |
| `dpo-r1/` | DPO Round 1 | loss 0.693→0.565 |
| `dpo-r2/` | DPO Round 2 | Conservative fine-tuning |
| `orpo/` | ORPO (experimental) | SFT+alignment simultaneous |
| `dpo-r3/` | DPO R3 (experimental) | Repetition-targeted |
| `data/` | Reproduction data | 684K preference + 105 repetition pairs |
| `configs/` | Training configs | SFT/DPO/ORPO YAMLs |
| `scripts/` | Training code | dpo.py, orpo_native.py, lora.py, etc. |

A **3-billion-parameter hybrid Mamba-2 + Transformer** language model implemented from scratch, inspired by the NVIDIA [Nemotron-H](https://arxiv.org/abs/2504.03624) architecture. Designed for Chinchilla-optimal pretraining over 60 hours on 7× NVIDIA B200 GPUs.

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Nemotron-Nano Architecture Fragmentation](#nemotron-nano-architecture-fragmentation)
- [Hardware Environment](#hardware-environment)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Technical Details](#technical-details)
- [1B → 3B Transition](#1b--3b-transition)
- [3B Hardware Constraint Optimization](#3b-hardware-constraint-optimization)
- [Training Data](#training-data)
- [Development History](#development-history)
- [SFT (Supervised Fine-Tuning)](#sft-supervised-fine-tuning)
- [Model Alignment & Evaluation](#model-alignment--evaluation)
  - [SFT Model Evaluation Results](#sft-model-evaluation-results)
  - [Preference Data Preparation](#preference-data-preparation)
  - [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
  - [Comprehensive Evaluation Results](#comprehensive-evaluation-results)
  - [ORPO Comparison Experiment (2026-03-25)](#orpo-comparison-experiment-2026-03-25)
  - [Deployment & Inference](#deployment--inference)
  - [Future Improvement Directions](#future-improvement-directions)
- [Appendix: Execution Guide](#appendix-execution-guide)
- [Benchmark Results](#benchmark-results)
- [Related Projects](#related-projects)
- [References](#references)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Project Overview

EVAFRILL-Mo is a project that implements a **hybrid SSM-Transformer** language model from scratch. Without relying on existing model hubs, every component — from the selective scan kernel to the training loop — is written directly in PyTorch.

**Key Features:**

- **Hybrid Mamba-2 + Transformer** layer architecture following the NVIDIA Nemotron-H design
- **Mamba-2 SSM** with custom selective scan and optional **SwiGLU FFN**
- **GQA (Grouped Query Attention)** for efficient sparse attention layers
- **FP8 native training** on B200 GPUs (MXFP8 block scaling)
- **Chunked Cross-Entropy** loss that reduces logits memory usage by 8×
- **Chinchilla-optimal** training: ~60 hours training a 3B model on ~55B tokens
- Custom **SentencePiece tokenizer** with 64K vocabulary supporting Korean, English, code, and math

---

## Architecture

### 3B Model Configuration (training complete)

```
vocab_size:        64,000
d_model:           3,072
n_layers:          26  (Mamba-2 ×24 + Attention ×2)
n_heads:           24
n_kv_heads:        8   (GQA ratio 3:1)
d_ffn:             9,216
mamba_d_ffn:       4,608  (SwiGLU FFN inside Mamba block)
mamba_d_state:     128
mamba_head_dim:    64
mamba_n_groups:    8
mamba_chunk_size:  256
max_seq_len:       4,096
Total parameters:  ~2,944M (2.94B)
```

### Previous 1B Model Configuration (experiments complete)

```
d_model: 2,048 | n_layers: 18 (16M+2A) | n_heads: 16 | n_kv_heads: 4
d_ffn: 5,504 | mamba_d_ffn: 3,072 | Total parameters: ~994M
```

### Hybrid Layer Layout

Transformer attention layers are sparsely placed at approximately the 1/2 point and at the end of the network, interspersed among Mamba-2 SSM blocks:

```
3B Layer Layout (26 layers):
Layer  0-11:  Mamba-2 SSM ×12  ──┐
Layer 12:     Attention (GQA)     │  First half
Layer 13-23:  Mamba-2 SSM ×11  ──┘
Layer 24:     Attention (GQA)        Second half
Layer 25:     Mamba-2 SSM ×1
```

### Design Principles

| Component | Design Choice | Rationale |
|-----------|--------------|-----------|
| SSM Backbone | Mamba-2 selective scan | Linear-time sequence modeling, efficient on long contexts |
| Sparse Attention | GQA with RoPE | Captures global dependencies that SSM may miss |
| Mamba FFN | Optional SwiGLU | Nemotron-H innovation; increases model capacity without changing the scan |
| Loss Function | Chunked Cross-Entropy | Reduces peak memory by computing logits in chunks |
| Precision | FP8 (MXFP8BlockScaling) | B200 native support, ~2× throughput over BF16 |
| Normalization | RMSNorm | Faster and more stable than LayerNorm |

---

## Nemotron-Nano Architecture Fragmentation

### What is "Architecture Fragmentation"?

NVIDIA's Nemotron-H/Nano is an architecture designed for 8B/4B scale, thousands of GPUs, and training on trillions of tokens. Reproducing it exactly is impossible in our environment (7× B200, 65 hours).

Instead, we **extracted (fragmented) only the core design principles** and scaled them down to fit our constrained hardware. This is what "architecture fragmentation" means.

### What Was Adopted vs. Skipped

| Nemotron-Nano Original | Our Implementation | Status |
|---|---|---|
| Mostly Mamba-2, few Attention (~9:1) | 16M + 2A (8:1 ratio), similarly configured | ✅ Adopted |
| Attention placed at 1/3 and 2/3 depth | Same evenly-spaced placement (18-layer: positions 6, 12) | ✅ Adopted |
| SwiGLU FFN added inside Mamba block | Implemented via `mamba_d_ffn` config field (0=disabled, backward-compatible) | ✅ Adopted |
| Multi-head SSM with grouped heads | `mamba_n_groups=8`, `mamba_head_dim=64` | ✅ Adopted |
| GQA (Grouped Query Attention) | `n_kv_heads=8` (ratio 3:1) | ✅ Adopted |
| FP8 native training | TransformerEngine MXFP8BlockScaling | ✅ Adopted |
| Large d_state (128) | `mamba_d_state=128` | ✅ Adopted |
| Chunk-based selective scan | `mamba_chunk_size=256` | ✅ Adopted |
| MoE (Mixture of Experts) | — | ❌ Skipped (negligible benefit at small scale) |
| Knowledge Distillation | — | ❌ Skipped (no teacher model available) |
| RLHF/DPO pipeline | Native DPO + LoRA (without TRL) | ✅ Adopted (Post-SFT) |
| 4B/8B scale | Scaled down to 2.94B | 🔄 Scaled |
| Training on trillions of tokens | 55B tokens (~1.34 epochs, Chinchilla 93%) | 🔄 Scaled |

### Concrete Architecture Selection Process

#### Stage 1: Initial 3B Design (Failed)

Initially, we attempted a scale close to Nemotron-Nano:

```
Initial Design: FRANKENSTALLM-H 3B
  d_model:     3072
  n_layers:    40 (Mamba-2 ×37 + Attention ×3)
  mamba_d_ffn: 4608
  n_groups:    8
  → Total ~4.44B parameters
```

**Problem discovered:** Within 65 hours, only **7%** of Chinchilla-optimal (20 × 4.44B = 88.8B tokens) was trainable. This would clearly result in a severely undertrained model. At this scale, approximately 930 hours (39 days) would be required.

#### Stage 2: Systematic Scale Search (5-Model Benchmark)

We designed 5 configs that preserved the Nemotron-H-style architecture while adjusting only `d_model` and `n_layers`. The following principles were maintained across all configs:
- Mamba:Attention ratio approximately 8–12:1
- Attention layers placed at 1/3 and 2/3 depth
- `mamba_d_ffn = 1.5 × d_model`
- `mamba_n_groups = 8`, `mamba_head_dim = 64`

```
5 candidate models:
  1B:   d=2048, 18L (16M+2A)  →  994M parameters
  1.5B: d=2048, 28L (26M+2A)  → 1.48B parameters
  2B:   d=2560, 24L (22M+2A)  → 1.94B parameters
  2.5B: d=2560, 32L (30M+2A)  → 2.53B parameters
  3B:   d=3072, 26L (24M+2A)  → 2.95B parameters
```

Each model was benchmarked for 20 steps on 7× B200 to measure actual throughput, then Chinchilla achievement rate was calculated.

#### Stage 3: Final Decision — 1B

**Chinchilla Scaling Law** (Hoffmann et al., 2022): For a fixed compute budget, "right-sized model + sufficient data" always beats "large model + insufficient data."

```
1B:   90,455 tok/s × 65h = 21.2B tokens  →  107% of Chinchilla 19.9B  ✅
1.5B: 59,107 tok/s × 65h = 13.8B tokens  →   47% of Chinchilla 29.6B  ❌
2B:   51,076 tok/s × 65h = 11.9B tokens  →   31% of Chinchilla 38.8B  ❌
```

The 1.5B model would only train on half the required tokens, performing **worse** than a fully trained model of the same size. The 1B was the only Chinchilla-optimal candidate.

#### The Meaning of Scaling Down

The reduction from 3B (4.44B parameters) → 1B (994M parameters) is not a simple compromise:

- **Fully trained 1B > Undertrained 3B**: According to Chinchilla scaling, when compute budget is fixed, fully training a smaller model outperforms undertrained larger models on all downstream tasks
- **Nemotron-H design principles are scale-independent**: Architecture choices such as the Mamba-Attention hybrid pattern, SwiGLU FFN, and GQA are equally valid at 1B scale
- **Experimental value**: After validating the architecture at small scale, the same design can be scaled up to 3B/7B once a larger compute budget is available

---

## Hardware Environment

| Item | Specification |
|------|--------------|
| **GPU** | 7× NVIDIA B200 (183 GB VRAM per GPU, ~1.28 TB total) |
| **System RAM** | 2.2 TB |
| **CUDA** | 13.0 |
| **Storage** | GPFS 20 TB (9 TB free) |
| **PyTorch** | 2.10.0a0+nv25.12 (NVIDIA custom build, B200-optimized) |
| **FlashAttention** | 2.7.4.post1+25.12 |

> **Warning:** PyTorch is an NVIDIA custom build (`nv25.12`). Reinstalling via `pip install torch` will break B200 optimizations — **do not reinstall**.

---

## Project Structure

```
EVAFRILL-Mo/
├── README.md                  # This file
├── CLAUDE.md                  # AI assistant instructions
│
├── model/                     # Model architecture
│   ├── config.py              # LMConfig dataclass (with __post_init__ validation)
│   ├── transformer.py         # LLM main model (hybrid layer dispatcher)
│   ├── mamba_block.py         # Mamba-2 SSM + optional SwiGLU FFN
│   ├── attention.py           # GQA attention with RoPE
│   ├── layers.py              # RMSNorm, SwiGLU, embeddings
│   └── lora.py                # LoRA adapter (Attention + Mamba layers)
│
├── train/                     # Training
│   ├── pretrain.py            # Pretraining entrypoint
│   ├── trainer.py             # Training loop (DDP, FP8, checkpointing)
│   ├── sft.py                 # Supervised fine-tuning (SFT)
│   ├── dpo.py                 # DPO preference learning (Native, LoRA)
│   ├── orpo.py                # ORPO preference optimization (TRL-based)
│   ├── orpo_native.py         # ORPO native implementation (no TRL, used for actual training)
│   └── utils.py               # Cosine scheduler, DDP setup, checkpoint utils
│
├── data/                      # Data pipeline
│   ├── dataset.py             # PackedDataset (memmap + MADV_WILLNEED hint)
│   ├── prepare.py             # Tokenization pipeline
│   ├── prepare_sft_data.py    # SFT data preparation
│   ├── filter_sft_v2.py       # SFT data quality filtering
│   ├── sft_dataset.py         # SFT conversational dataset
│   ├── dpo_dataset.py         # DPO preference pair dataset
│   ├── prepare_preference_combined.py  # 7 preference sources → unified JSONL
│   ├── generate_repetition_preference.py  # Repetition-suppression preference data generation
│   └── *.bin                  # Binary token files (not included in repo)
│
├── eval/                      # Evaluation
│   ├── evafrill_eval.py       # Comprehensive 4-phase evaluation (PPL, generation, calibration, lm-eval)
│   ├── full_eval_pipeline.py  # Full evaluation pipeline orchestration
│   ├── perplexity.py          # Perplexity evaluation
│   ├── generate.py            # Text generation / sampling
│   ├── comprehensive_eval.py  # Comprehensive evaluation tool
│   └── report_generator.py    # Markdown evaluation report generation
│
├── scripts/                   # Launch, monitoring, and deployment scripts
│   ├── merge_checkpoints.py   # SLERP/LERP checkpoint interpolation (mitigates alignment tax)
│   ├── export_to_hf.py        # HuggingFace Hub model export + push
│   ├── convert_to_hf.py       # Native → HuggingFace format conversion
│   └── migrate_qkv_checkpoint.py  # QKV checkpoint layout migration
│
├── configs/                   # YAML training configuration files
├── benchmarks/                # Throughput & profiling tools
├── tokenizer/                 # SentencePiece tokenizer training
├── reports/                   # Evaluation and analysis reports
├── docs/                      # Hardware & environment documentation
├── train_3b_sft_1gpu.sh       # H100 MIG SFT launch script
├── train_3b_dpo_1gpu.sh       # H100 MIG DPO launch script
├── train_3b_orpo_1gpu.sh      # H100 MIG ORPO launch script
├── requirements.txt           # Python dependencies
├── README.en.md               # English README
└── demo/app.py                # Gradio demo server
```

---

## Quick Start

### Prerequisites

```bash
# Install required libraries (PyTorch is pre-installed — do not reinstall)
pip install transformers accelerate peft trl deepspeed bitsandbytes sentencepiece wandb
```

### Single GPU Test

```bash
python train/pretrain.py \
    --config configs/small.yaml \
    --train_data data/train.bin \
    --batch_size 8
```

### Multi-GPU Training — 3B Model (7× B200, FP8)

```bash
torchrun --nproc_per_node=7 train/pretrain.py \
    --config /tmp/bench_3b.yaml \
    --train_data data/3b_train.bin \
    --batch_size 6 \
    --lr 3e-4 \
    --warmup_steps 6395 \
    --max_steps 319772 \
    --use_fp8
```

### Auto-Restart Training (automatic recovery on crash)

```bash
nohup bash train_3b_resilient.sh &
```

### Training Monitoring

```bash
# Training log (loss, tok/s, lr per step)
tail -F checkpoints/3b_final/train.log

# Restart / error event monitor
tail -F checkpoints/3b_final/monitor.log
```

### Inference Example (Python)

```python
import torch
from model.transformer import LLM
from tokenizers import Tokenizer

# Load model (SLERP checkpoint recommended)
model = LLM.from_pretrained("checkpoints/3b_dpo/checkpoint-slerp")
model = model.to(device="cuda:0", dtype=torch.bfloat16)
model.eval()

tok = Tokenizer.from_file("tokenizer/korean_sp/tokenizer.json")

# Apply chat template
prompt = "<|user|>\nWhat is artificial intelligence?\n<|assistant|>\n"
ids = torch.tensor([tok.encode(prompt).ids], device="cuda:0")

# Generation (recommended: temp=0.7, rep_penalty=1.2)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(256):
        logits, _ = model(ids)
        logits = logits[:, -1, :].float()
        # Repetition penalty
        for prev_id in set(ids[0].tolist()):
            if logits[0, prev_id] > 0: logits[0, prev_id] /= 1.2
            else: logits[0, prev_id] *= 1.2
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == 2: break  # EOS

print(tok.decode(ids[0].tolist()))
```

> 💡 **Gradio Demo**: Run `python3 demo/app.py` and visit http://localhost:7860
>
> 📦 **HuggingFace**: Download model from [pathcosmos/EVAFRILL-Mo-3B](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

### Download from HuggingFace and Run Inference

> **GGUF/Ollama not supported**: The Mamba-2 hybrid architecture is incompatible with llama.cpp/GGUF format. Only PyTorch direct inference is available.

**Step 1: Clone source code** (custom architecture modules required)

```bash
git clone https://github.com/pathcosmos/EVAFRILL-Mo
cd EVAFRILL-Mo
```

**Step 2: Download checkpoints** (HuggingFace Hub, SLERP recommended)

```bash
git lfs install
git clone https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B

# Required files: slerp/config.json (687B), slerp/model.safetensors (5.9GB), slerp/tokenizer.json (4.2MB)
```

**Step 3: Install dependencies**

```bash
pip install torch safetensors tokenizers PyYAML
# Optional (GPU acceleration): pip install mamba_ssm causal_conv1d
```

**Step 4: Run inference** (direct safetensors loading)

```python
import json
import torch
from model.config import LMConfig
from model.transformer import LLM
from tokenizers import Tokenizer
from safetensors.torch import load_file as load_safetensors

CKPT = "path/to/EVAFRILL-Mo-3B/slerp"

# Load config
with open(f"{CKPT}/config.json") as f:
    data = json.load(f)
for k in ("model_type", "architectures", "_variant", "_description"):
    data.pop(k, None)
cfg = LMConfig(**data)
cfg.use_flash_attn = False  # inference compatibility

# Load model
model = LLM(cfg)
state = load_safetensors(f"{CKPT}/model.safetensors", device="cpu")
model.load_state_dict(state, strict=False)
model = model.to(device="cuda:0", dtype=torch.bfloat16)
model.eval()

# Tokenizer
tok = Tokenizer.from_file(f"{CKPT}/tokenizer.json")

# Generate
prompt = "<|user|>\nWhat is artificial intelligence?\n<|assistant|>\n"
ids = torch.tensor([tok.encode(prompt).ids], device="cuda:0")

with torch.no_grad():
    for _ in range(256):
        logits, _ = model(ids)
        logits = logits[:, -1, :].float()
        for prev_id in set(ids[0].tolist()):
            if logits[0, prev_id] > 0: logits[0, prev_id] /= 1.2
            else: logits[0, prev_id] *= 1.2
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tok.token_to_id("</s>"): break

print(tok.decode(ids[0].tolist()))
```

**Alternative: Use the evaluation framework runner**

The `evafrill_runner.py` in [frankenstallm_test](https://github.com/pathcosmos/frankenstallm_test) wraps the above process into a simple API:

```python
from eval_framework.evafrill_runner import generate, unload_model

result = generate("Hello, please introduce yourself.")
print(result["response"])
print(f"Speed: {result['tokens_per_sec']:.1f} TPS")

unload_model()  # free VRAM
```

> See the [frankenstallm_test README](https://github.com/pathcosmos/frankenstallm_test#evafrill-mo-모델-설정-pytorch-직접-추론) for setup instructions.

**System Requirements**

| Item | Minimum | Recommended |
|------|---------|-------------|
| GPU VRAM | 8 GB (BF16) | 16 GB+ |
| RAM | 16 GB | 32 GB |
| CPU inference | Possible (~0.5 TPS) | GPU recommended (~4.8 TPS) |

---

## Technical Details

A complete reference of the core techniques applied in this project.

### SSM / Mamba-2

| Technique | Description | Location |
|-----------|-------------|----------|
| **Triton Chunked SSD Kernel** | `mamba_chunk_scan_combined` from `mamba_ssm` — a Triton-written chunked Structured State Space Duality kernel. Memory-efficient O(N) sequence processing | `model/mamba_block.py:333` |
| **causal_conv1d** | Fused CUDA kernel handling causal depthwise conv1d + SiLU activation in a single kernel | `model/mamba_block.py:312` |
| **Selective Scan (pure PyTorch fallback)** | Pure PyTorch selective scan implementation for environments without CUDA kernels. Chunk-based for memory efficiency | `model/mamba_block.py:54` |
| **Multi-head SSM** | Grouped SSM with 64 heads divided into 8 groups. Core structure of Mamba-2 | `mamba_n_groups=8`, `mamba_head_dim=64` |
| **A_log Parameterization** | Diagonal decay matrix A learned in log space for numerical stability. `exp(-exp(A_log) * dt)` | `model/mamba_block.py:219` |
| **dt_bias Initialization** | Time-step bias initialized as `log(uniform(0.001, 0.1))` for early training stability | `model/mamba_block.py:227` |
| **Mamba SwiGLU FFN** | SwiGLU FFN added inside Mamba block in Nemotron-H style. Disabled when `mamba_d_ffn=0` (backward-compatible) | `model/mamba_block.py` |

### Transformer / Attention

| Technique | Description | Location |
|-----------|-------------|----------|
| **FlashAttention-2** | Tri Dao's IO-aware attention algorithm. Exact attention computation in O(N) memory | `model/attention.py:211` |
| **GQA (Grouped Query Attention)** | 24 query heads, 8 KV heads (3:1 ratio). 67% reduction in KV cache memory | `model/attention.py:77` |
| **RoPE (Rotary Positional Embedding)** | Rotary positional encoding for relative position information. `rope_theta=500000` | `model/layers.py:54`, `model/attention.py:39` |
| **RMSNorm** | Reduced computation vs. LayerNorm (no mean calculation). Pre-norm architecture | `model/layers.py:27` |
| **SwiGLU FFN** | Shazeer (2020) SwiGLU gated activation. `gate * silu(up)` structure | `model/layers.py:109` |

### Precision / Quantization

| Technique | Description | Location |
|-----------|-------------|----------|
| **FP8 (MXFP8BlockScaling)** | TransformerEngine Microscaling FP8. Utilizes B200's FP8 tensor cores for ~2× throughput over BF16 | `train/trainer.py:163` |
| **fp8_autocast** | Hybrid precision: TE modules (te.Linear) compute in FP8, rest remain in BF16 | `train/trainer.py:470` |
| **BF16 autocast** | `torch.autocast(dtype=bfloat16)` — pure PyTorch layers (Mamba) auto-cast to BF16 | `train/trainer.py:467` |
| **te.Linear (FP8 Linear)** | TransformerEngine FP8 Linear applied to QKV/Output projections in attention layers | `model/attention.py:103` |
| **FP8 Alignment Validation** | `__post_init__` verifies `d_model`, `d_ffn`, `mamba_d_ffn` are all multiples of 16 | `model/config.py:120` |

### Loss Function / Memory Optimization

| Technique | Description | Location |
|-----------|-------------|----------|
| **Chunked Cross-Entropy** | Computes logits (B×T×V) in chunks rather than all at once. 8× logits memory reduction with 64K vocabulary | `model/transformer.py:232` |
| **Gradient Accumulation + no_sync** | Uses `model.no_sync()` during accumulation steps in DDP to prevent unnecessary allreduce | `train/trainer.py:243` |
| **gradient_as_bucket_view** | DDP gradient buffers used directly as NCCL communication buckets. Eliminates memory copies (zero-copy) | `train/pretrain.py:323` |

### Distributed Training / Hardware Optimization

| Technique | Description | Location |
|-----------|-------------|----------|
| **DDP (DistributedDataParallel)** | Data-parallel training across 7× B200 GPUs. NCCL backend | `train/pretrain.py:317` |
| **NUMA Affinity** | GPU 0–3 → NUMA node 0 (cores 0–35), GPU 4–6 → NUMA node 1 (cores 36–71). 3.2× reduction in memory access latency | `train/pretrain.py:256` |
| **DistributedSampler** | Evenly distributes data across GPUs to prevent duplicate training | `train/pretrain.py:335` |
| **expandable_segments** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — prevents CUDA memory fragmentation | Environment variable |

### Data Pipeline

| Technique | Description | Location |
|-----------|-------------|----------|
| **np.memmap** | Memory-maps training data for direct disk reads. Maps 82 GB of data without loading fully into RAM | `data/dataset.py:38` |
| **MADV_RANDOM** | Informs the kernel of random access patterns to disable unnecessary read-ahead | `data/dataset.py:95` |
| **MADV_WILLNEED** | Asynchronously prefaults pages into the page cache | `data/dataset.py:96` |
| **persistent_workers** | Keeps DataLoader workers alive across epochs. Eliminates worker re-creation overhead | `train/pretrain.py:355` |
| **pin_memory** | Page-pinned memory for CPU→GPU transfers. Accelerates DMA transfers | `train/pretrain.py:352` |
| **prefetch_factor=4** | Pre-loads 4 batches per worker to minimize GPU wait time | `train/pretrain.py:354` |
| **6 workers/GPU** | 6×7=42 workers, balanced with OMP_NUM_THREADS=4 within 72-core CPU budget | `train/pretrain.py:351` |

### Training Stability / Scheduling

| Technique | Description | Location |
|-----------|-------------|----------|
| **Cosine LR Schedule + Linear Warmup** | Learning rate adjusted with cosine decay after warmup. `min_lr_ratio=0.1` (final lr = 3e-5) | `train/utils.py:35` |
| **AdamW (selective weight decay)** | bias, RMSNorm, A_log, D, and dt_bias parameters excluded from weight decay | `train/pretrain.py:203` |
| **Gradient Clipping (max_norm=1.0)** | L2-norm-based gradient clipping. Prevents gradient spikes in Mamba | `train/trainer.py:280` |
| **NaN Detection + Emergency Checkpoint** | Immediately saves checkpoint and emits warning upon detecting NaN/Inf during training | `model/mamba_block.py:349` |
| **Auto-Restart Wrapper** | Automatically restarts from the latest checkpoint on crash. Auto-increments port number (prevents EADDRINUSE) | `train_1b_resilient.sh` |

### Tokenizer

| Technique | Description | Location |
|-----------|-------------|----------|
| **SentencePiece BPE** | Byte-Pair Encoding with 64K vocabulary. Mixed training on Korean + English + code + math | `tokenizer/` |
| **HuggingFace-Compatible Conversion** | Converts SentencePiece model to HF tokenizer format | `tokenizer/convert_sp_to_hf.py` |

---

## 1B → 3B Transition

### Discovery: tok/s Was Per-GPU

After starting 1B model training, we detected that progress was much faster than expected.

```
~1 hour after 1B training started:
  step 3,700 / 45,776 (8.1%)
  elapsed: 0.8 hours
  estimated completion: ~9.3 hours
```

**Cause: Misinterpretation of the throughput metric.** The `tokens_per_sec` calculation in `trainer.py` was a local (per-GPU) value:

```python
# trainer.py:335 — batch_size is the local (per-GPU) batch
tokens_per_sec = (batch_size * seq_len * grad_accum * log_interval) / elapsed
```

That is, `tok/s 90,000` in the log was the throughput of **a single GPU**, and the true aggregate throughput was:

```
Actual aggregate: 90,000 × 7 GPUs = 630,000 tok/s
```

### Recalculation: 1B Needs Only 1/7 of 65 Hours

| Item | Previous Calculation (Wrong) | Corrected Calculation |
|------|:---:|:---:|
| tok/s | 90,000 (aggregate) | 630,000 (aggregate) |
| Tokens in 65h | 21.1B | **147.4B** |
| Chinchilla achievement | 107% | **751%** |
| Actual time required | ~64.8h | **~8.8h** |

**Investing 65 hours in the 1B model would mean training at 7.5× Chinchilla — severe over-training.** This implies a large remaining compute budget, making it possible to train a much larger model.

### Decision to Switch to 3B

With the corrected calculations, the full model scale was re-evaluated:

| Model | tok/s (agg) | Tokens in 60h | Chinchilla | Achievement |
|-------|----------:|--------:|---------:|------:|
| 1B | 630,000 | 136.1B | 20B | 681% (over) |
| 1.5B | 367,213 | 79.3B | 30B | 264% (over) |
| 2B | 271,894 | 58.7B | 38B | 155% (over) |
| 2.5B | 260,519 | 56.3B | 50B | 113% |
| **3B** | **254,681** | **55.0B** | **58.9B** | **93%** |

**3B is the largest model that can achieve 93% of Chinchilla within the 60-hour budget.** The in-progress 1B training (step 4,230) was halted and switched to 3B.

---

## 3B Hardware Constraint Optimization

### Core Constraint: Mamba Memory Cliff

During the 3B benchmark, **OOM occurred when going from batch size 6 to 7**. This is because the Mamba-2 Triton Chunked SSD kernel fully materializes intermediate tensors (intermediate states) at a certain threshold.

```
3B model batch size test results (7× B200, FP8):
  batch=6  →  47.3 GB/GPU  ✅ (stable)
  batch=7  →  OOM          ❌ (Memory Cliff)
  batch=8  →  OOM          ❌
  batch=10 →  OOM          ❌
  batch=12 →  OOM          ❌
```

**Cliff mechanism:** The `mamba_chunk_scan_combined` kernel allocates intermediate tensors of shape `(batch, n_chunks, n_heads, chunk_size, d_state)`. Up to batch=6, it streams these chunk-by-chunk, but from batch=7 onward, it materializes everything in memory at once, causing an explosion from **47 GB → 183 GB+**.

### Optimized 3B Training Configuration

Settings that maximize throughput at the maximum batch size below the cliff (batch=6):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **batch_size** | 6 (per-GPU) | Maximum value just before Memory Cliff. 47.3 GB / 183 GB |
| **grad_accum** | 1 | Additional accumulation yields no throughput gain (same wall clock) |
| **effective_batch** | 42 seqs (172,032 tok) | 6 × 7 GPUs × 4,096 seq_len |
| **lr** | 3e-4 | Standard learning rate for 3B scale |
| **warmup_steps** | 6,395 | 2% of total steps (prevents excessive initial gradients) |
| **max_steps** | 319,772 | 55B tokens / 172,032 tok/step |
| **weight_decay** | 0.1 | AdamW standard (excluding bias, norm, SSM parameters) |
| **precision** | FP8 (MXFP8BlockScaling) | ~2× throughput over BF16 |
| **max_grad_norm** | 1.0 | Prevents Mamba gradient spikes |
| **min_lr_ratio** | 0.1 | Final lr = 3e-5 |
| **seed** | 42 | Reproducibility |

### Throughput Analysis

```
3B model measured performance:
  per-GPU:    36,383 tok/s
  aggregate:  254,681 tok/s (×7 GPUs)
  step time:  ~0.67s/step
  GPU memory: 47.3 GB / 183 GB (25.8% used)
  GPU util:   nearly 100% (compute-bound)
```

### Memory Efficiency Analysis

At batch=6, only 25.8% of GPU memory is used, yet due to the Mamba Memory Cliff, batch=7 exceeds 183 GB. This "unused" 74.2% of VRAM cannot be utilized due to structural constraints of Mamba SSM.

```
Memory breakdown (estimated):
  Model weights (FP8):    ~3.0 GB
  Optimizer states:       ~18.0 GB (AdamW, FP32 moments)
  Gradient buffers:       ~6.0 GB
  Activations (batch=6):  ~20.3 GB
  ──────────────────────────────
  Total:                  ~47.3 GB
```

### Data Constraints

| Item | Value |
|------|-------|
| Training data | 41.1B tokens (82 GB) |
| Processable in 60h | 55.0B tokens |
| Epochs required | ~1.34 |
| Chinchilla achievement | ~93% (1 epoch: 70%, 1.34 epochs: 93%) |

1.34 epochs of data repetition is acceptable — the Chinchilla paper itself permits 1–2 epochs of data repetition, and recent research (Muennighoff et al., 2023) shows negligible performance degradation up to 4 epochs.

### Automatic Recovery System

`train_3b_resilient.sh` for 60-hour continuous training stability:

```
Recovery mechanism:
  1. Crash detection (exit code ≠ 0)
  2. Force-kill GPU processes + wait for memory release
  3. Auto-locate latest checkpoint (checkpoint-XXXXXXX)
  4. Auto-increment port number (prevents EADDRINUSE)
  5. Wait 30 seconds, then restart
  6. Maximum 10 retries
```

---

## Training Data

| Item | Value |
|------|-------|
| **Total Token Count** | ~41.1B (82 GB binary) |
| **Training Usage** | ~55B tokens (3B model, ~1.34 epochs) |
| **Tokenizer** | Custom SentencePiece, 64K vocabulary |
| **Supported Languages** | Korean, English, code, math |

### Data Sources

| Source | Domain |
|--------|--------|
| Cosmopedia | Web text, stories, textbooks |
| Korean C4 | Korean web crawl |
| Korean Wikipedia | Korean encyclopedia |
| Namu Wiki | Korean wiki |
| CC-100 Korean | CommonCrawl Korean subset |
| MathPile | Mathematical text |
| OpenWebMath | Web-based math data |
| HPLT Korean | High Performance Language Technology data |

### Training Hyperparameters (3B Main Training)

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| LR schedule | Cosine decay (min_lr_ratio=0.1) |
| Warmup steps | 6,395 (2% of total steps) |
| Total steps | 319,772 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Batch size | 6 per GPU (42 total) — Memory Cliff constraint |
| Sequence length | 4,096 |
| Precision | FP8 (MXFP8BlockScaling) |
| Throughput | ~36,383 tok/s (per-GPU), ~254,681 tok/s (aggregate) |
| Estimated duration | ~60 hours |
| Chinchilla achievement | ~93% |

### Previous 1B Training Hyperparameters (Experimental)

| Parameter | Value |
|-----------|-------|
| Batch size | 16 per GPU (112 total) |
| Total steps | 45,776 |
| Throughput | ~90,000 tok/s (per-GPU), ~630,000 tok/s (aggregate) |
| Actual duration | ~8.8 hours (interrupted; switched to 3B at step 4,230) |

---

## Development History

EVAFRILL-Mo is the result of an iterative design journey through 6 major stages.

### Stage 1 — [FRANKENSTALLM](https://github.com/pathcosmos/FRANKENSTALLM) (Pure Transformer)

Started with a pure Transformer decoder-only LLM (Frankenstein + LLM). Trained a custom SentencePiece tokenizer on Korean + English + code + math data (vocabulary 64,000), and built the foundational training pipeline (DDP, checkpointing, cosine scheduler). The full code and documentation for that project are available at the [FRANKENSTALLM GitHub repository](https://github.com/pathcosmos/FRANKENSTALLM).

### Stage 2 — 11-Step Implementation Plan (Fully Completed)

1. **Config Validation** — `__post_init__` divisibility checks in the `LMConfig` dataclass
2. **Mamba FFN Integration** — Optional SwiGLU, backward-compatible (`mamba_d_ffn=0` disables it)
3. **NaN Detection** — Emergency checkpoint save upon NaN detection during training
4. **CUDA Kernel Optimization** — Selective scan performance optimization
5. **Chunked Cross-Entropy** — 1/8 reduction in logits memory (critical with 64K vocabulary)
6. **FP8 Training** — TransformerEngine MXFP8BlockScaling on B200
7. **Gradient Clipping & Monitoring** — `max_grad_norm=1.0`, gnorm tracking
8. **Checkpoint Save/Restore** — Full DDP compatibility, including optimizer/scheduler state
9. **Cosine LR Schedule** — Linear warmup + cosine decay (`min_lr_ratio=0.1`)
10. **Data Pipeline Optimization** — Memmap + `MADV_WILLNEED` + persistent workers
11. **Multi-GPU DDP** — Distributed training across 7× B200

### Stage 3 — Nemotron-Nano Architecture Fragmentation & Optimal Scale Search (EVAFRILL-Mo)

Core question: **What is the largest model that can achieve Chinchilla-optimal training in 65 hours × 7 B200?**

- Extracted core design principles from Nemotron-Nano and applied them to 5 scales (1B–3B) (details: [Architecture Fragmentation section](#nemotron-nano-architecture-fragmentation))
- Systematic benchmark of 5 models (20 steps each, 7 GPUs)
- **Mamba Memory Cliff phenomenon discovered**: ~7.5× memory jump at batch size threshold
- **1B model selected as final choice**: Only Chinchilla-optimal candidate (107% achievement)

### Stage 4 — VectorDB / Memory DB Investigation

Investigated whether VectorDB or memoryDB would benefit LLM pretraining:

| Approach | Findings | Decision |
|----------|----------|----------|
| RETRO-style retrieval-augmented training | Not applicable to Mamba — CCA layers are Transformer-specific architecture | ❌ Not applicable |
| LMDB/RocksDB data loading | 82 GB data fully cached in 2.2 TB RAM → no improvement | ❌ Unnecessary |
| Curriculum Learning (DB-based) | Possible without DB; ~1–3% improvement level | ❌ DB unnecessary |
| FAISS/Milvus/LanceDB | Not installed; introduction overhead too high | ❌ Cost exceeds benefit |

**Conclusion:** Under the 65-hour deadline, implementation overhead would eat into training time; not recommended. Best to focus on pure pretraining.

### Stage 5 — 1B Training Start & Overtraining Detection

- **Model**: 994M parameters, 18 layers (Mamba-2 ×16 + Attention ×2)
- **Training started**: 45,776 steps, batch=16, ~90,000 tok/s (per-GPU)
- **Detection**: At step 3,700, total estimated time was ~9.3 hours
- **Root cause analysis**: Confirmed tok/s was per-GPU → actual aggregate is 630,000 tok/s
- **Judgment**: 65 hours on 1B = 7.5× Chinchilla over-training → wasted compute
- **Decision**: Stopped 1B training at step 4,230; switched to 3B scale

### Stage 6 — 3B Pretraining Completed

- **Model**: 2,944M parameters, 26 layers (Mamba-2 ×24 + Attention ×2)
- **Benchmark**: Sequential testing from batch=6–12; batch=6 was the maximum before Memory Cliff
- **Throughput**: 36,383 tok/s (per-GPU), 254,681 tok/s (aggregate)
- **Training**: 319,772 steps, ~55B tokens, ~60 hours
- **Chinchilla achievement**: ~93% (1.34 epochs)
- **Checkpoints**: Auto-saved every 1,000 steps (model + optimizer + scheduler + train_state)
- **Recovery wrapper**: `train_3b_resilient.sh` — auto-restarts from latest checkpoint on crash (up to 10 retries, auto port change)
- **Completed**: 2026-03-09, all 319,772 steps finished. Final checkpoint: `checkpoints/3b_final/checkpoint-0319772`

#### Pretraining Loss Trend (25k-interval average)

| Interval | Avg Loss | Change |
|----------|--------:|--------|
| 0–25k | 2.96 | Initial convergence |
| 25–50k | 4.77 | Epoch transition spike |
| 50–100k | 2.39 | Rapid decrease |
| 100–150k | 2.00 | Steady decrease |
| 150–200k | 1.87 | Gradual decrease |
| 200–250k | 1.77 | Gradual decrease |
| 250–319k | 1.69 | Convergence complete |

### Stage 7 — 3B SFT v2 (Completed with Early Stop)

Performed Korean SFT (Supervised Fine-Tuning) on top of the pretrained 3B model.

#### Environment Transition: B200 8GPU → H100 MIG 1GPU

After returning the B200 cluster, transitioned to an H100 MIG 3g.40gb single-partition environment.

| Item | B200 8GPU (Pretraining) | H100 MIG (SFT) |
|------|------------------------|-----------------|
| GPU | 8× B200 (183 GB each) | 1× H100 MIG 3g.40gb (~42 GB) |
| Precision | FP8 (MXFP8) | BF16 + Gradient Checkpointing |
| Batch | bs=6 × 7 GPU = 42 | bs=4, grad_accum=7, eff=28 |
| Speed | 0.67 s/step | 6.8 s/step |

#### SFT Training Configuration

| Parameter | Value |
|-----------|-------|
| Base checkpoint | `checkpoints/3b_final/checkpoint-0319772` |
| SFT data | `data/sft_combined/train_filtered.jsonl` |
| Validation data | `data/sft_combined/val_filtered.jsonl` |
| Config file | `configs/h100_mig/korean_3b_sft_1gpu.yaml` |
| Launch script | `train_3b_sft_1gpu.sh` (resilient wrapper) |
| batch_size | 4 |
| grad_accum_steps | 7 |
| effective batch | 28 |
| max_steps | 135,000 |
| eval_interval | 5,000 steps |
| lr | 7.0e-06 (cosine decay) |
| warmup_steps | 500 |
| weight_decay | 0.01 |
| max_grad_norm | 1.0 |
| NEFTune alpha | 5.0 |
| Precision | BF16 + Gradient Checkpointing |
| VRAM usage | 24.0 GB / 40.3 GB (60%) |
| Tokenization | Full pre-tokenize + cache at initialization |

#### SFT Validation Loss Trend — Convergence and Early Stop Rationale

| Step | val_loss | Δval_loss | Phase |
|-----:|--------:|---------:|-------|
| 5,000 | 1.8774 | — | Rapid decrease |
| 10,000 | 1.8424 | -0.0350 | |
| 15,000 | 1.8239 | -0.0185 | |
| 20,000 | 1.8124 | -0.0115 | Deceleration |
| 25,000 | 1.8050 | -0.0074 | |
| 30,000 | 1.8001 | -0.0049 | |
| 35,000 | 1.7968 | -0.0033 | |
| 40,000 | 1.7949 | -0.0019 | Plateau entry |
| 45,000 | 1.7940 | -0.0009 | |
| 50,000 | 1.7933 | -0.0007 | |
| 55,000 | 1.7928 | -0.0005 | |
| 60,000 | 1.7928 | -0.0000 | Stagnation |
| **65,000** | **1.7924** | **-0.0004** | **Early Stop decision** |

13 consecutive best updates, but improvements after 50K dropped to measurement noise level.

#### Early Stop Decision (Step 65,000 / 135,000, 48.15%)

**Decision date**: 2026-03-22
**Final best val_loss**: 1.7924 (step 65,000)
**Final checkpoint**: `checkpoints/3b_sft_v2/checkpoint-best`, `checkpoint-0065059` (emergency)

**Stop rationale — mathematical analysis:**

1. **Asymptote reached**: Exponential decay fitting (`L = a·exp(-b·t) + c`) gives theoretical minimum val_loss (c) ≈ 1.7922. Current value of 1.7924 is already nearly at the asymptote (R² = 0.9994)
2. **Improvement exhausted**: 50K→65K (15,000 steps, ~28 hours) total improvement: 0.0009. Expected improvement over remaining 70K steps (~5.5 days): 0.001–0.003
3. **PPL difference negligible**: val_loss difference of 0.001 = PPL 6.006 → 6.000 (ΔPPL = 0.006). Imperceptible in actual output quality
4. **Insufficient SNR**: Expected improvement (0.0002) vs. measurement noise per 5K-step interval (σ = 0.0003) → SNR = 0.57σ — not statistically significant

**Stop rationale — practical analysis:**

1. **Opportunity cost**: The same GPU time could yield much higher expected return through quantitative evaluation (KoBEST/KLUE), data restructuring + new SFT, or DPO/RLHF
2. **No overfitting**: val–train gap remained stable at 0.01–0.03 across all intervals; no monotonic increase
3. **Cosine LR tail effect exhausted**: LR already at 53% of peak; unlikely to see sharp improvement in the later phase

#### SFT Training Stability Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Max gnorm | 4.219 (warmup step 140) | Normal |
| gnorm > 5 | 0 occurrences | Safe |
| nan/inf/OOM | 0 occurrences | Safe |
| Memory | 24.0 GB fixed throughout | Stable |
| tok/s trend | Average 5,343; no decrease over time | Stable |
| SIGTERM recovery | 1 occurrence at step 421, resumed normally | Normal |
| epoch | 0 (single epoch, no data repetition) | Normal |

---

## SFT (Supervised Fine-Tuning)

### Overview

Performed SFT on the pretrained 3B model (`checkpoints/3b_final/checkpoint-0319772`) using Korean instruction-following data. Conducted on a single H100 MIG 3g.40gb GPU; convergence analysis led to early stop at step 65,000.

### SFT Data

| Item | Value |
|------|-------|
| Training data | `data/sft_combined/train_filtered.jsonl` |
| Validation data | `data/sft_combined/val_filtered.jsonl` |
| Format | Conversational JSONL |
| Tokenization | Full pre-tokenize + `.sft_cache_*.pt` cache at initialization |

### Key Techniques

| Technique | Description |
|-----------|-------------|
| **NEFTune** (alpha=5.0) | Injects uniform noise into embeddings to improve generalization (Jain et al., 2023) |
| **Dynamic Padding** | Pads to the maximum sequence length in the batch, aligned to 64. Reduces wasted computation vs. fixed-length padding |
| **Gradient Checkpointing** | Recomputes activations to save VRAM. Enables 3B model training within the MIG 42 GB constraint |
| **Cosine LR Decay** | Cosine decay from peak 7.0e-06. Conservative setting at 1/43 of the pretraining lr (3e-4) |
| **Resilient Wrapper** | `train_3b_sft_1gpu.sh` — auto checkpoint save and restart on SIGTERM/crash |

### Results Summary

```
Training period:  2026-03-17 ~ 2026-03-22 (5 days)
Steps completed:  65,000 / 135,000 (48.15%)
Final val_loss:   1.7924 (13 consecutive best updates)
Stop reason:      Plateau — asymptote reached; expected return from further training < measurement noise
Checkpoint:       checkpoints/3b_sft_v2/checkpoint-best (step 65,000)
```

### Convergence Visualization

```
val_loss
1.880 ┤ ●
      │  ╲
1.860 ┤   ╲
      │    ╲
1.840 ┤     ●
      │      ╲
1.820 ┤       ●
      │        ╲
1.800 ┤         ●──●
      │              ╲
1.795 ┤               ●──●──●──●──●──●  ← Plateau
      │
1.790 ┤─────────────────────────────────
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──→ step (×1000)
         5  10 15 20 25 30 35 40 45 50 55 60 65
```

- **Rapid decrease** (5K–20K): val_loss 1.877 → 1.812, Δ = -0.065
- **Deceleration** (20K–35K): val_loss 1.812 → 1.797, Δ = -0.015
- **Plateau** (35K–65K): val_loss 1.797 → 1.792, Δ = -0.005 (improvement at noise level)

---

## Model Alignment & Evaluation

After SFT v2 completion (step 65,000), this section covers model quality evaluation and alignment via DPO (Direct Preference Optimization).

### SFT Model Evaluation Results

Completed Phase 2 (generation quality) of the 4-phase evaluation framework using `eval/evafrill_eval.py`. Phases 1, 3 were skipped (low priority / time constraints); Phase 4 (lm-eval) was aborted after 9 hours because `kmmlu` alone contains 269 subtasks (~167,000 problems), making the estimated runtime 12–18 hours on a single H100 MIG GPU — GPU time was reallocated to DPO training.

**Evaluation environment**: H100 MIG 3g.40gb, batch_size=2

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 (PPL) | Perplexity on 3b_val.bin | ⏭ Skipped (~4.4h, low priority) |
| Phase 2 (Generation) | 15 prompts × 4 decoding configs | ✅ Completed (~2.5h) |
| Phase 3 (Calibration) | Calibration curve | ⏭ Skipped |
| Phase 4 (lm-eval) | 6 benchmarks (kmmlu, etc.) | ❌ Aborted (9h) |

**Phase 2 Generation Quality Results (checkpoint-best, step 65,059):**

| Prompt | Greedy 3-gram Repetition Rate | Assessment |
|--------|------------------------------:|-----------|
| 대한민국의 수도는 | 96.85% | Same-phrase repetition loop |
| 양자 컴퓨터란 | 96.85% | Severe repetition |
| 건강한 식습관을 위해서는 | 59.45% | Relatively acceptable |
| 인공지능이란 | 50.00% | Structured list but repetition present |
| 한국어는 세계에서 | 35.83% | Low repetition, Korean/English mixed corruption |
| **Average** | **~76%** | **DPO needed to resolve repetition** |

**Key findings:** SFT model generates Korean text, but severe repetition loops occur under greedy decoding. Repetition penalty (1.2) improves output but is not a fundamental fix — **preference learning via DPO is essential**.

### Preference Data Preparation

Used `data/prepare_preference_combined.py` to merge 7 Korean preference datasets into a unified JSONL.

| Dataset | Record Count | Format |
|---------|-------------:|--------|
| heegyu/orca-math-korean-preference-cleaned | 192,422 | chosen/rejected |
| nayohan/preference-collection-ko-full | 199,577 | orig_response_A/B + orig_preference |
| kuotient/orca-math-word-problems-193k-korean | 192,375 | chosen/rejected |
| FreedomIntelligence/alpaca-gpt4-korean | 49,969 | chosen/rejected |
| heegyu/orca_ko | 42,989 | chosen/rejected |
| HAERAE-HUB/KOFFQA-GuardInstruct-v1 | 7,210 | chosen/rejected |
| jojo0217/korean_rlhf_dataset | 0 | SFT-only (no preference pairs) |
| **Total** | **684,542 → 504,103** | Valid samples after tokenization |

### DPO (Direct Preference Optimization)

#### DPO vs ORPO: Method Comparison & Selection Rationale

Both DPO and ORPO align the model using "chosen vs rejected" preference pairs, but differ in implementation and training stage.

| | DPO | ORPO |
|---|-----|------|
| **Reference model** | Required (logprob of SFT model) | **Not required** |
| **VRAM** | High (additional ref model forward pass) | Low |
| **Loss function** | `log σ(β · (Δchosen - Δrejected))` | SFT loss + λ · odds ratio penalty |
| **Training stage** | SFT → DPO (2 stages) | **Simultaneous with SFT** (1 stage) |
| **Maturity** | Standard, widely validated | Relatively new (2024) |

**Reasons for choosing DPO:**
1. **SFT is already complete** — ORPO's advantage is SFT+alignment simultaneously, but SFT v2 already converged at step 65,000; restarting would waste 5 days
2. **VRAM disadvantage resolved via LoRA B-zeroing** — Temporarily zero lora_B to compute ref logprob; operates at 6.3 GB without model duplication
3. **Nemotron-H paper uses DPO** — The architectural reference uses 2-round DPO + SLERP merge; same strategy followed here

> **Note:** If designing from scratch, ORPO could be more efficient by combining SFT + alignment in one pass. `train/orpo.py` already exists in the project for future experiments.

#### Training Configuration

**Design decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Framework** | Native DPO (no TRL) | TRL requires HF AutoModel — not supported for Hybrid Mamba |
| **Parameter efficiency** | LoRA (rank=32, alpha=64) | ~22 GB VRAM → fits H100 MIG 42 GB with room to spare |
| **Reference model** | LoRA B-zeroing | Temporarily zero lora_B to compute ref logprob; no model duplication |
| **Checkpoint merging** | SLERP interpolation | Nemotron-H style: `slerp(W_sft, W_dpo, α=0.5)` to mitigate alignment tax |

**LoRA adapter configuration:**

```
Applied layers:    Attention (qkv_proj, out_proj) + Mamba-2 (in_proj, out_proj)
Number of adapters: 52
Trainable params:  21,438,464 (0.72% of total 2.97B)
VRAM usage:        ~6.3 GB (15% of MIG 42 GB)
```

**2-Round DPO Strategy (Nemotron-H style):**

- **Round 1 (Exploration)**: Learns broad preference signals from the full 504K dataset. Higher β (0.1) and lr (5e-7) allow fast exploration of the preference direction.
- **Round 2 (Exploitation)**: Fine-tunes on top of Round 1's merged checkpoint with lower β (0.05) and lr (1e-7). Lowering β reduces deviation from the reference model, **preventing over-alignment** while preserving SFT fluency.

| | Round 1 | Round 2 |
|---|---------|---------|
| **Purpose** | Broad preference learning (exploration) | Fine-tuning (exploitation) |
| **Data** | Full preference set (504K samples) | Same or high-quality subset |
| **Steps** | 3,000 | 2,000 |
| **Beta** | 0.1 | 0.05 (prevents over-alignment) |
| **LR** | 5e-7 | 1e-7 (10× lower) |
| **Warmup** | 100 steps | 50 steps |
| **Batch** | bs=1 × grad_accum=16 = eff 16 | Same |

#### Training Results

**Round 1** (2026-03-23, 4h 33m, 6.3 GB VRAM):

```
  step   10 | loss 0.6941 | margin -0.006 | lr 5.0e-08  (warmup)
  step  100 | loss 0.6855 | margin  0.006 | lr 5.0e-07  (warmup complete)
  step  500 | loss 0.6543 | margin  0.120 | lr 4.93e-07
  step 1500 | loss 0.6012 | margin  0.210 | lr 2.50e-07
  step 2500 | loss 0.5717 | margin  0.280 | lr 7.50e-08
  step 3000 | loss 0.5652 | margin  0.245 | lr 5.0e-08   (final)
  → Checkpoint: checkpoints/3b_dpo_r1/checkpoint-0003000
```

Loss 0.693 → 0.565 (18.5% decrease); margin +0.245 — model clearly learned to distinguish chosen from rejected. Stable throughout: gnorm < 5, no NaN.

**Round 2** (2026-03-23, 3h 2m, 6.3 GB VRAM):

```
  step   50 | loss 0.6953 | margin  0.003 | lr 1.0e-07  (warmup complete)
  step  500 | loss 0.6880 | margin  0.027 | lr 8.9e-08
  step 1000 | loss 0.6906 | margin  0.008 | lr 5.7e-08
  step 1500 | loss 0.6884 | margin  0.017 | lr 2.5e-08
  step 2000 | loss 0.6886 | margin -0.005 | lr 1.0e-08  (final)
  → Checkpoint: checkpoints/3b_dpo_r2/checkpoint-0002000
```

Loss 0.692 → 0.689 (0.5% change): intentionally gradual — low β (0.05) and lr (1e-7) prevent over-alignment. gnorm 1.6–2.2 (more stable than Round 1's 3–4).

#### SLERP Merge & Final Model Selection

**What is alignment tax?** During DPO, the model learns preference alignment but partially loses SFT knowledge and fluency. SLERP mitigates this.

**SLERP (Spherical Linear Interpolation)** merges two checkpoints via spherical interpolation in weight space. Unlike LERP, SLERP preserves the direction of weight vectors:

```
SLERP(W_sft, W_dpo, α=0.5):
  α=0: Pure SFT (repetition issues remain)
  α=0.5: 50% SFT + 50% DPO (Nemotron-H default)
  α=1: Pure DPO (maximum alignment tax)
```

**3-checkpoint comparison** (SFT vs DPO R2 vs SLERP α=0.5) on 15 prompts, greedy decoding (2026-03-24):

| Prompt | SFT | DPO R2 | SLERP | Best |
|--------|----:|-------:|------:|------|
| 대한민국의 수도는 | 85.0 | 89.4 | 96.9 | SFT |
| 인공지능이란 | 61.8 | 61.8 | **50.0** | SLERP |
| 한국의 전통 음식 중에서 | 90.9 | 74.8 | **39.4** | SLERP |
| 지구 온난화의 주요 원인은 | 82.3 | 87.4 | **72.4** | SLERP |
| 프로그래밍을 배우려면 | 89.0 | 89.0 | 90.6 | SFT/DPO |
| 조선시대에는 | 65.0 | 84.3 | **65.0** | SFT=SLERP |
| 물리학에서 에너지란 | 88.6 | 93.7 | **86.6** | SLERP |
| 한국어는 세계에서 | 65.8 | 65.8 | **52.0** | SLERP |
| 경제 성장을 위해서는 | 77.2 | 77.2 | **70.5** | SLERP |
| 우주 탐사의 역사를 보면 | 95.3 | 95.3 | 95.3 | Tied |
| 머신러닝과 딥러닝의 차이는 | 89.4 | 89.4 | **83.1** | SLERP |
| 한국 문학의 대표적인 작품으로는 | 74.0 | **72.8** | 85.4 | DPO |
| 양자 컴퓨터란 | 96.9 | 96.9 | 96.9 | Tied |
| 건강한 식습관을 위해서는 | 56.3 | **55.9** | 55.9 | DPO=SLERP |
| 세계 2차 대전 이후 | 79.5 | 77.6 | 77.6 | DPO=SLERP |
| **Average** | **79.8%** | **80.7%** | **74.5%** | **SLERP** |

| Model | Avg repetition | Prompts with lowest repetition |
|-------|:-------------:|:------------------------------:|
| SFT v2 | 79.8% | 1/15 |
| DPO Round 2 | 80.7% | 1/15 |
| **SLERP (α=0.5)** | **74.5%** | **7/15** |

**Final model selected: SLERP (α=0.5)** — `checkpoints/3b_dpo/checkpoint-slerp`

Rationale: lowest repetition in 7/15 prompts; "한국의 전통 음식" 90.9% → 39.4% (-51.5pp). Limitations: still far from 30% target (74.5%); 2 prompts regressed vs SFT; DPO-only was marginally worse than SFT (80.7% vs 79.8%). Root cause appears to be an architecture-level issue — greedy decoding repetition in hybrid Mamba-3B may have inherent limits.

### Comprehensive Evaluation Results

#### Generation Quality Comparison (Greedy Repetition)

Combined assessment across Phase 2 repetition and Phase 4 accuracy (limit=100):

| Model | Repetition (↓) | lm-eval Accuracy (↑) | Overall |
|-------|:--------------:|:-------------------:|---------|
| SFT | 79.8% | 28.3% | Baseline |
| DPO R2 | 80.7% | 28.3% | Repetition worse, knowledge retained |
| **SLERP** | **74.5%** | **28.3%** | **Best repetition, same knowledge → Final** |

**lm-eval 3-way comparison (limit=100, kmmlu excluded, 0-shot):**

| Benchmark | SFT | DPO R2 | SLERP | Note |
|-----------|----:|-------:|------:|------|
| hellaswag | 39.0% | 39.0% | 39.0% | Identical |
| belebele_kor_Hang | 30.0% | 29.0% | 30.0% | SFT=SLERP |
| arc_easy | 28.0% | 28.0% | 27.0% | |
| arc_challenge | 21.0% | 22.0% | 22.0% | |
| global_mmlu_full_ko | 23.4% | 23.4% | 23.3% | Nearly identical |

Accuracy difference across all three checkpoints is within 1% — **alignment tax is negligible**. LoRA-based DPO + SLERP effectively preserves knowledge.

#### Repetition Penalty Decoding Test

Applying `repetition_penalty=1.2` at inference on the SLERP model **dramatically reduced repetition**:

| Prompt | greedy (r=1.0) | greedy (r=1.2) | t0.7 + r1.2 |
|--------|:--------------:|:--------------:|:-----------:|
| 대한민국의 수도는 | 81.5% | **13.4%** | **0.4%** |
| 인공지능이란 | 61.8% | **13.4%** | **1.6%** |
| 한국의 전통 음식 중에서 | 74.8% | **0.0%** | **0.0%** |
| 건강한 식습관을 위해서는 | 66.1% | **0.8%** | **1.2%** |
| 한국어는 세계에서 | 48.0% | 0.0% | 0.0% |
| **Average** | **66.4%** | **~5.5%** | **~0.6%** |

**Generation quality examples (greedy + r=1.2):**

```
[대한민국의 수도는] → 서울특별시이고, 그 외 지역은 광역시로 분류한다.
  대한민국의 행정구역 변천사 1945년 8월 15일 - 경기도 인천부(仁川府)
  1949년 7월 14일 - 경기도 인천시(仁川市)...

[한국의 전통 음식 중에서] → 가장 유명한 것이 바로 김치이다. 김치는
  한국인의 주식이자, 세계인에게 사랑받는 국민음식으로 자리 잡았다.
  김치가 세계적으로 유명해진 이유는 무엇일까? 그 비밀은 바로 '배추'에 있다...

[건강한 식습관을 위해서는] → 균형 잡힌 식단이 중요하다. 특히, 단백질은
  필수 아미노산으로 구성돼 있어 체내 흡수율이 높아 건강에 좋다...
```

**Conclusion**: DPO/SLERP alone reduced repetition only to 74.5%, but **applying rep_penalty=1.2 at inference drops it to ~5%**. **Recommended inference settings**: `temperature=0.7, repetition_penalty=1.2` (t0.7_r1.2).

#### lm-eval Benchmarks

**SLERP final model (limit=500, kmmlu excluded, 0-shot):**

| Benchmark | Accuracy | Random | Note |
|-----------|:--------:|:------:|------|
| hellaswag | **34.6%** | 25.0% | English commonsense +9.6pp |
| arc_easy | **32.0%** | 25.0% | Basic science +7.0pp |
| global_mmlu_full_ko | 23.7% | 25.0% | Korean domain knowledge weak |
| belebele_kor_Hang | 23.6% | 25.0% | Korean reading comprehension limited |
| arc_challenge | 18.2% | 25.0% | Advanced reasoning insufficient |

Limited by 3B model scale — only hellaswag/arc_easy show meaningful improvement over random. Korean benchmarks remain at random level — a fundamental constraint of model size and training data volume.

#### Multi-α SLERP Experiment

Compared SLERP at α=0.3, 0.5, 0.7 on the same 5 prompts with greedy + rep_penalty=1.2 (2026-03-24):

| α | SFT weight | DPO weight | Avg repetition |
|---|-----------|-----------|----------------|
| 0.3 | 70% | 30% | 9.1% |
| **0.5** | **50%** | **50%** | **5.8%** (lowest) |
| 0.7 | 30% | 70% | 7.3% |

**Conclusion**: α=0.5 (Nemotron-H default) is optimal. A 50:50 balance between SFT and DPO is most effective at suppressing repetition.

#### Qualitative Chat Template Evaluation

Tested SLERP (α=0.5) with the SFT chat template (`<|user|>\n...\n<|assistant|>\n`). Decoding: `temperature=0.7, repetition_penalty=1.2` (2026-03-24).

```
USER: 김치를 만드는 방법을 간단히 알려주세요.
ASSISTANT: 김치는 한국 요리에서 중요한 역할을 하는 음식으로, 녹색 채소가
  주재료인 발효 식품이랍니다! 이 음식은 발효 과정을 통해 배추와 무에
  영양분을 공급하여 건강과 맛의 조화를 추구하는데요...

USER: 인공지능이 무엇인지 쉽게 설명해주세요.
ASSISTANT: 인공지능(artificial intelligence, AI)은 기계가 스스로 학습하고,
  의사 결정 능력을 가지고 있는 컴퓨터의 한 종류로, 인간의 지능을 모방하는
  것을 목표로 합니다...

USER: 건강을 위해 어떤 운동을 하면 좋을까요?
ASSISTANT: 건강에 좋은 운동은 여러 가지가 있습니다. 먼저, 심폐 지구력과
  근력 향상을 돕는 실내 자전거 타기와 수영이 추천됩니다. 또한, 필라테스와
  요가도 훌륭한 운동이 될 것입니다...
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Repetition** | ✅ Good | Virtually none with rep_penalty=1.2 |
| **Fluency** | ✅ Good | Natural Korean sentence generation |
| **Instruction following** | ⚠️ Fair | Attempts to answer but accuracy is low |
| **Factual accuracy** | ❌ Poor | Hallucinations present |
| **Code generation** | ❌ Failed | Meaningless response to Fibonacci request |

**Summary**: Limited by 3B scale — fluent Korean generation is achievable, but factual accuracy and complex reasoning remain weak.

### ORPO Comparison Experiment (2026-03-25)

DPO failed to directly solve repetition (SFT 79.8% → DPO 80.7%, worsened). ORPO learns SFT+alignment simultaneously — we tested whether it overcomes the structural limitations of the separated pipeline.

#### What is ORPO and Why Native Implementation?

ORPO (Odds Ratio Preference Optimization, Hong et al., 2024) combines SFT loss and preference loss in one objective:

```
L_ORPO = L_SFT + λ * L_OR
  L_SFT: CrossEntropy on chosen response
  L_OR:  -log(σ(log(odds_chosen / odds_rejected)))
```

| | DPO | ORPO |
|---|---|---|
| Reference model | Required | **Not needed** |
| Training stages | SFT → DPO (2 stages) | **1 stage from pretrained** |

Existing `train/orpo.py` uses TRL → requires HF AutoModel → incompatible with custom Mamba-2 hybrid. Native implementation was written (`train/orpo_native.py`), same reason as DPO.

#### Training Configuration & Results

| Item | Value |
|------|-------|
| Starting point | `checkpoints/3b_final/checkpoint-0319772` (Pretrained) |
| Data | 504,103 preference pairs (same as DPO) |
| Steps | 10,000 |
| LR | 5e-6 (10× DPO — starting from pretrained) |
| λ (OR weight) | 1.0 |
| LoRA | rank=32, alpha=64 |
| VRAM | 6.2 GB |
| Duration | 12h 48m |

```
Training trajectory:
  step     10 | sft 10.16 | or 0.909 | total 11.07  (start)
  step  1,000 | sft  6.25 | or 0.751 | total  7.00
  step  5,000 | sft  6.03 | or 0.565 | total  6.60
  step 10,000 | sft  5.85 | or 0.558 | total  6.41  (final)
```

SFT loss -42.4%, OR loss -38.6%.

#### Head-to-Head Comparison

| Metric | SLERP (α=0.5) | ORPO (10K) | Winner |
|--------|:-------------:|:----------:|:------:|
| **Greedy repetition** | **74.5%** | 87.1% | SLERP |
| **greedy+r1.2 repetition** | 5.5% | **3.7%** | ORPO |
| **t0.7+r1.2 repetition** | **0.6%** | 1.8% | SLERP |
| **hellaswag** | **39.0%** | 35.0% | SLERP |
| **arc_easy** | 27.0% | **30.0%** | ORPO |
| **belebele_kor** | **30.0%** | 23.0% | SLERP |
| **arc_challenge** | 22.0% | **19.0%** | SLERP |
| **global_mmlu_ko** | 23.3% | 23.3% | Tied |
| **Chat quality** | ✅ Fluent | ❌ Broken | **SLERP** |
| **Training time** | 5d+8h | **12.8h** | ORPO |

#### Analysis and Conclusion

**SLERP wins (under current settings).** Key reason for ORPO's weakness: insufficient SFT learning — ORPO's SFT loss stopped at 5.85 vs SFT v2's final val_loss of 1.79. 10,000 ORPO steps is far fewer than SFT's 65,000 steps, causing broken chat responses and higher greedy repetition. rep_penalty=1.2 slightly favors ORPO (3.7% vs 5.5%) — OR loss does contribute to repetition suppression.

**For a fair comparison**, ORPO needs 65,000+ steps (~5 days). Current 10,000 steps is an exploratory experiment. ORPO's time efficiency (12.8h vs 5d+8h) is attractive, but OR loss alignment only manifests after SFT loss converges sufficiently. The SLERP pipeline provides more stable results for this model/data combination.

### Deployment & Inference

**Model download**: [🤗 pathcosmos/EVAFRILL-Mo-3B](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

**Gradio demo server:**
```bash
python3 demo/app.py  # http://localhost:7860
```

**GGUF/Ollama conversion — currently not possible:**

This model uses a **custom hybrid Mamba-2 + Transformer** architecture, making llama.cpp-based GGUF/Ollama conversion impossible.

| Tool | Support | Reason |
|------|---------|--------|
| **llama.cpp/GGUF** | ❌ No | Only experimental pure Mamba-2 (CPU only), hybrid unsupported |
| **Ollama** | ❌ No | Built on llama.cpp, same limitations |
| **vLLM** | ⚠️ Theoretically | Supports Mamba2ForCausalLM, but requires custom weight key mapping (days of work) |
| **Gradio (pure Python)** | ✅ Running | `demo/app.py` |

**Technical barriers:**
- No standardized way to manage SSM state (Mamba) + KV cache (Attention) simultaneously in GGUF
- `mamba_ssm` CUDA kernels not implemented in llama.cpp
- llama.cpp only supports static layer types — hybrid dispatch not possible
- NVIDIA Nemotron-H (same architecture family) faces the same GGUF conversion issues ([llama.cpp #20570](https://github.com/ggml-org/llama.cpp/issues/20570))

> **Note:** This is a deliberate tradeoff of choosing a custom hybrid architecture — performance and research flexibility over portability. The model can be served via vLLM or the pure Python inference server.

### Repetition-Targeted DPO Experiment (DPO Round 3, 2026-03-25)

#### Motivation

Existing DPO used general preference data (504K) but failed to directly solve repetition (SFT 79.8% → DPO 80.7%). Testing whether **explicit repetitive/non-repetitive pairs** enable DPO to directly target repetition.

#### Self-Generated Preference Data

Generated two decodings for the same prompts using the SLERP model:
- **rejected**: greedy (temp=0, rep_penalty=1.0) → repetitive (avg 71.7%)
- **chosen**: sampling (temp=0.7, rep_penalty=1.2) → clean (avg 0.1%)

105 preference pairs from 105 Korean prompts (10 categories: daily life, science, history, career, health, creative writing, tech, culture, environment, etc.) via `data/generate_repetition_preference.py`. Combined with existing 504K for 684,647 total pairs.

#### Training Configuration & Results

| Item | Value |
|------|-------|
| Starting point | `checkpoints/3b_dpo/checkpoint-slerp` (SLERP final model) |
| Data | 684,647 pairs (504K existing + 105 repetition-targeted) |
| Steps | 1,000 |
| Beta | 0.05 |
| LR | 1e-7 |
| VRAM | 6.3GB |
| Duration | ~1.5 hours |

```
Training trajectory:
  step   10 | loss 0.6932 | margin -0.007
  step  100 | loss 0.6888 | margin +0.013
  step  500 | loss 0.6925 | margin +0.014
  step 1000 | loss 0.6910 | margin +0.014  (final)
```

Minimal loss change (0.693→0.691). The model was already well-aligned via SLERP, so additional training has small effect. The 105 repetition-targeted samples are diluted within 684K (0.015%).

Checkpoint: `checkpoints/3b_dpo_r3/checkpoint-merged`

#### Evaluation Results

**Greedy repetition comparison (15-prompt average):**

| Model | Greedy repetition | rep_penalty=1.2 (5p) |
|-------|:-----------------:|:-------------------:|
| SLERP (α=0.5) | **74.5%** | **5.8%** |
| DPO R3 (repetition-targeted) | 79.4% | 4.5% |

**Per-prompt detail (greedy + rep_penalty=1.2):**

| Prompt | SLERP r1.2 | R3 r1.2 |
|--------|:----------:|:-------:|
| 대한민국의 수도는 | 13.4% | **0.4%** |
| 인공지능이란 | **13.4%** | 13.8% |
| 한국의 전통 음식 | 0.0% | 0.0% |
| 건강한 식습관 | **0.8%** | 7.5% |
| 프로그래밍을 배우려면 | 1.6% | **0.8%** |

#### Analysis and Conclusion

**DPO R3 shows no significant improvement over SLERP.**

- Greedy repetition: SLERP 74.5% → R3 79.4% (**actually worsened**)
- rep_penalty=1.2: SLERP 5.8% → R3 4.5% (marginal improvement)
- **Root cause**: 105 repetition-targeted pairs are only 0.015% of 684K — too diluted to affect behavior
- **Lesson**: Self-generated preference data needs thousands to tens of thousands of pairs minimum. ~100 pairs are buried in 684K existing data

---

### Future Improvement Directions

1. ~~Repetition-targeted preference data~~ → ✅ Experiment completed (see above)
2. **Scale up repetition data** — Expand from 105 to thousands/tens of thousands of pairs for DPO retraining
3. **SFT data quality audit** — Investigate hallucination and garbled output root causes
4. **Scale up** — Move to 7B+ models with larger compute budget

---

## Appendix: Execution Guide

### DPO Pipeline Commands

```bash
# DPO Round 1 + Round 2 + SLERP Merge full pipeline
bash train_3b_dpo_1gpu.sh

# Or run individually
python3 train/dpo.py \
    --sft_checkpoint checkpoints/3b_sft_v2/checkpoint-best \
    --dpo_data data/preference/combined_preference.jsonl \
    --config configs/h100_mig/dpo_3b_1gpu.yaml \
    --device cuda:0

# SLERP checkpoint merging
python3 scripts/merge_checkpoints.py \
    --ckpt_a checkpoints/3b_sft_v2/checkpoint-best \
    --ckpt_b checkpoints/3b_dpo_r1/checkpoint-merged \
    --output checkpoints/3b_dpo/checkpoint-slerp \
    --alpha 0.5
```

### Log Monitoring

```bash
# DPO training step-wise loss/margin/lr
tail -f /root/taketimes/llm/EVAFRILL-Mo/checkpoints/3b_dpo_r1/train.log

# Full stdout (model loading, data parsing included)
tail -f /root/taketimes/llm/EVAFRILL-Mo/checkpoints/3b_dpo_r1/stdout.log
```

### Bug Fix History

- **LoRA device mismatch fix** (`model/lora.py`): `lora_A`/`lora_B` parameters in `LoRALinear.__init__` were created on CPU, causing device mismatch with the original layer on GPU. Fixed by using `original.weight.device`/`dtype` to create them on the same device.
- **nayohan preference parser added** (`data/prepare_preference_combined.py`): Added support for datasets in `orig_response_A/B + orig_preference` format (previously parsed 0 records).

---

## Benchmark Results

### Chinchilla Feasibility by Model Scale (60 hours, 7× B200)

> **Note:** tok/s values are **per-GPU**. Multiply by ×7 for total (aggregate) throughput.

| Model | Parameters | tok/s (per-GPU) | tok/s (agg ×7) | Max Batch | Memory/GPU | 60h Tokens | Chinchilla | Achievement |
|:------|----------:|----------------:|---------------:|----------:|-----------:|-----------:|-----------:|------------:|
| 1B | 994M | 90,000 | 630,000 | 16 | 16.0 GB | 136.1B | 19.9B | 681% |
| 1.5B | 1.48B | 52,459 | 367,213 | 12 | 23.7 GB | 79.3B | 29.6B | 268% |
| 2B | 1.94B | 38,842 | 271,894 | 10 | 31.0 GB | 58.7B | 38.8B | 151% |
| 2.5B | 2.53B | 37,217 | 260,519 | 6 | 40.5 GB | 56.3B | 50.6B | 111% |
| **3B** | **2.94B** | **36,383** | **254,681** | **6** | **47.3 GB** | **55.0B** | **58.9B** | **93%** ✅ |

> **Conclusion:** Given that tok/s is per-GPU, 1B–2.5B models greatly exceed Chinchilla within 60 hours (overtraining). **3B is the optimal scale that most efficiently fits the compute budget at ~93% Chinchilla.**

### Mamba Memory Cliff Phenomenon

An important phenomenon discovered during benchmarking: Mamba-2's selective scan exhibits a **dramatic memory cliff** at a specific batch size threshold.

```
Based on the 1.5B model:
  batch 12 → 23.7 GB/GPU
  batch 16 → 178  GB/GPU  (7.5× increase!)
```

This occurs because the selective scan fully materializes intermediate states in memory when the product of batch size, sequence length, and state dimension exceeds an internal chunking boundary. The key factors are `mamba_chunk_size=256` and `d_state=128`.

---

## Related Projects

- **[FRANKENSTALLM](https://github.com/pathcosmos/FRANKENSTALLM)** | [🤗 HuggingFace](https://huggingface.co/pathcosmos/frankenstallm) — The predecessor to EVAFRILL-Mo. A project that began as a pure Transformer decoder-only LLM. Built foundational infrastructure including a custom Korean+English+code+math tokenizer and DDP training pipeline. EVAFRILL-Mo evolved from this into a hybrid Mamba-2 + Transformer architecture.

A 3B hybrid model implemented from scratch, inspired by the NVIDIA Nemotron-H architecture. While FRANKENSTALLM is pure Transformer-based, EVAFRILL-Mo adopts a Mamba-2 SSM + sparse Transformer attention hybrid structure.

| Item | FRANKENSTALLM | EVAFRILL-Mo |
|------|---------------|-------------|
| Architecture | Pure Transformer (28L) | Mamba-2 24L + Attention 2L |
| Parameters | 3.17B | 2.94B |
| Key techniques | GQA, FP8, FlashAttention-2 | Selective Scan, SwiGLU FFN in Mamba, GQA |
| Design principle | Proven Transformer architecture | Nemotron-H fragmentation |
| GPUs | 8× B200 | 7× B200 |
| Training strategy | Chinchilla-optimal | Chinchilla 93% target |

Both projects share the same tokenizer (64K SentencePiece), training data pipeline, and DDP/FP8 infrastructure — "same ingredients, different recipe" — enabling a controlled comparison of how architecture differences affect performance.

---

## References

| Paper | Authors | Key Contribution |
|-------|---------|-----------------|
| [Nemotron-H](https://arxiv.org/abs/2504.03624) | NVIDIA, 2025 | Hybrid Mamba-Transformer architecture design |
| [Mamba-2: Structured State Space Duality](https://arxiv.org/abs/2405.21060) | Dao & Gu, 2024 | SSD (Structured State Space Duality) algorithm |
| [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) | Gu & Dao, 2023 | Original Selective State Space Model |
| [Chinchilla Scaling Law](https://arxiv.org/abs/2203.15556) | Hoffmann et al., 2022 | Optimal compute allocation — tokens = 20× params |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Tri Dao, 2023 | IO-aware attention, O(N) memory |
| [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245) | Ainslie et al., 2023 | KV-cache-efficient attention |
| [SwiGLU Activation](https://arxiv.org/abs/2002.05202) | Shazeer, 2020 | Gated activation function |
| [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | Su et al., 2021 | Relative positional encoding |
| [Scaling Data-Constrained LMs](https://arxiv.org/abs/2305.16264) | Muennighoff et al., 2023 | Effect of repeated training data (up to 4 epochs) |
| [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | Rafailov et al., 2023 | Preference alignment without reward models |
| [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691) | Hong et al., 2024 | Unified SFT + preference optimization in a single stage |
| [NEFTune](https://arxiv.org/abs/2310.05914) | Jain et al., 2023 | Embedding noise injection for fine-tuning quality improvement |

---

## Acknowledgments

This project was conducted using GPU computing resources provided through the **"Advanced GPU Utilization Support Program"** (MSIT Notice No. 2025-1068) by the **Ministry of Science and ICT (MSIT)** of the Republic of Korea.

> **National AI Computing Resource Support Portal**: [https://aiinfrahub.kr](https://aiinfrahub.kr)
>
> - Organized by: Ministry of Science and ICT (MSIT), National IT Industry Promotion Agency (NIPA)
> - Operated by: Korea Association of Information & Telecommunication (KAIT)

We are deeply grateful for the national-level AI computing infrastructure support from the Korean government, which made it possible to train a Korean 3B hybrid Mamba-Transformer model from scratch on 7× NVIDIA B200 GPUs.

- **NVIDIA Nemotron-H** — Inspiration for the hybrid Mamba-Transformer architecture design
- **Mamba-2** (Dao & Gu, 2024) — Foundation for the structured state space model
- **Chinchilla Scaling Law** (Hoffmann et al., 2022) — Criterion for optimal training compute allocation
- **Technologies used:** PyTorch, FlashAttention-2, TransformerEngine
- **[FRANKENSTALLM](https://github.com/pathcosmos/FRANKENSTALLM)** — Foundation project

---

## License

This project is distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

*EVAFRILL-Mo — Built from scratch, one selective scan at a time.*

</div>

> **[한국어](README.md)** | English
