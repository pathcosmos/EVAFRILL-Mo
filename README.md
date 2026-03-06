<div align="center">

# EVAFRILL-Mo

**Hybrid Mamba-2 + Transformer Language Model**

*Bride Eva (Frankenstein's bride) + FRIDAY (Iron Man's AI) + LLM + Nemotron's Mo*

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10.0--nv25.12-EE4C2C?logo=pytorch&logoColor=white)
![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![FlashAttention 2](https://img.shields.io/badge/FlashAttention-2.7.4-blueviolet)
![FP8](https://img.shields.io/badge/FP8-Native-orange)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![GPUs](https://img.shields.io/badge/GPUs-7%C3%97%20B200-76B900?logo=nvidia&logoColor=white)

A **1B-parameter hybrid Mamba-2 + Transformer** language model built from scratch, inspired by NVIDIA's [Nemotron-H](https://arxiv.org/abs/2504.03624) architecture. Designed for Chinchilla-optimal pretraining on 7× NVIDIA B200 GPUs.

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Hardware Environment](#-hardware-environment)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training Data](#-training-data)
- [Development History](#-development-history)
- [Benchmark Results](#-benchmark-results)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🔭 Project Overview

EVAFRILL-Mo is a ground-up implementation of a **hybrid SSM-Transformer** language model. Rather than relying on existing model hubs, every component — from the selective scan kernel to the training loop — is written from scratch in PyTorch.

**Key highlights:**

- **Hybrid Mamba-2 + Transformer** layers in a single model, following NVIDIA's Nemotron-H design
- **Mamba-2 SSM** with custom selective scan and optional **SwiGLU FFN** per block
- **Grouped Query Attention (GQA)** for efficient sparse attention layers
- **FP8 native training** (MXFP8 block scaling) on B200 GPUs
- **Chunked Cross-Entropy** loss reducing logits memory consumption to 1/8
- **Chinchilla-optimal** training: 1B model trained on ~21B tokens in ~65 hours
- Custom **SentencePiece tokenizer** (64K vocab) covering Korean, English, Code, and Math

---

## 🏗 Architecture

### 1B Model Configuration

```
vocab_size:        64,000
d_model:           2,048
n_layers:          18  (16 Mamba-2 + 2 Attention)
n_heads:           16
n_kv_heads:        4   (GQA ratio 4:1)
d_ffn:             5,504
mamba_d_ffn:       3,072  (SwiGLU FFN in Mamba blocks)
mamba_d_state:     128
mamba_head_dim:    64
mamba_n_groups:    8
mamba_chunk_size:  256
max_seq_len:       4,096
total_params:      ~994M
```

### Hybrid Layer Layout

The model interleaves Mamba-2 SSM blocks with sparse Transformer attention layers placed at roughly 1/3 and 2/3 positions through the network:

```
Layer  0–7:   Mamba-2 SSM  ──┐
Layer  8:     Attention (GQA) │  Block 1
Layer  9–16:  Mamba-2 SSM  ──┘
Layer  17:    Attention (GQA)    Block 2
```

### Design Principles

| Component | Design Choice | Rationale |
|-----------|--------------|-----------|
| SSM backbone | Mamba-2 selective scan | Linear-time sequence modeling, efficient for long contexts |
| Sparse attention | GQA with RoPE | Captures global dependencies that SSM may miss |
| Mamba FFN | Optional SwiGLU | Nemotron-H innovation; adds capacity without changing scan |
| Loss function | Chunked Cross-Entropy | Reduces peak memory by computing logits in chunks |
| Precision | FP8 (MXFP8BlockScaling) | B200-native, ~2× throughput vs BF16 |
| Normalization | RMSNorm | Faster and more stable than LayerNorm |

---

## 🖥 Hardware Environment

| Item | Specification |
|------|--------------|
| **GPU** | 7× NVIDIA B200 (183 GB VRAM each, ~1.28 TB total) |
| **System RAM** | 2.2 TB |
| **CUDA** | 13.0 |
| **Storage** | GPFS 20 TB (9 TB free) |
| **PyTorch** | 2.10.0a0+nv25.12 (NVIDIA custom build, B200-optimized) |
| **FlashAttention** | 2.7.4.post1+25.12 |

> ⚠️ **Warning:** PyTorch is an NVIDIA custom build (`nv25.12`). Do **not** reinstall via `pip install torch` — this will break B200 optimizations.

---

## 📁 Project Structure

```
EVAFRILL-Mo/
├── README.md                  # This file
├── CLAUDE.md                  # AI assistant instructions
│
├── model/                     # Model architecture
│   ├── config.py              # LMConfig dataclass with __post_init__ validation
│   ├── transformer.py         # LLM main model (hybrid layer dispatcher)
│   ├── mamba_block.py         # Mamba-2 SSM + optional SwiGLU FFN
│   ├── attention.py           # GQA attention with RoPE
│   └── layers.py              # RMSNorm, SwiGLU, embeddings
│
├── train/                     # Training
│   ├── pretrain.py            # Main pretraining entry point
│   ├── trainer.py             # Training loop (DDP, FP8, checkpointing)
│   ├── sft.py                 # Supervised fine-tuning
│   ├── orpo.py                # ORPO preference optimization
│   └── utils.py               # Cosine scheduler, DDP setup, checkpoint utils
│
├── data/                      # Data pipeline
│   ├── dataset.py             # PackedDataset (memmap + MADV_WILLNEED hints)
│   ├── prepare.py             # Tokenization pipeline
│   └── *.bin                  # Binary token files (not tracked in repo)
│
├── eval/                      # Evaluation
│   ├── perplexity.py          # Perplexity evaluation
│   ├── generate.py            # Text generation / sampling
│   └── comprehensive_eval.py  # Full evaluation suite
│
├── configs/                   # YAML training configurations
├── scripts/                   # Launch, monitor, and deploy scripts
├── benchmarks/                # Throughput & profiling tools
├── tokenizer/                 # SentencePiece tokenizer training
├── reports/                   # Evaluation & analysis reports
├── docs/                      # Hardware & environment documentation
└── train_1b_resilient.sh      # Auto-restart training wrapper (crash recovery)
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install required libraries (PyTorch is pre-installed — do NOT reinstall)
pip install transformers accelerate peft trl deepspeed bitsandbytes sentencepiece wandb
```

### Single GPU Test

```bash
python train/pretrain.py \
    --config configs/small.yaml \
    --train_data data/train.bin \
    --batch_size 8
```

### Multi-GPU Training (7× B200 with FP8)

```bash
torchrun --nproc_per_node=7 train/pretrain.py \
    --config /tmp/bench_1b.yaml \
    --train_data data/3b_train.bin \
    --batch_size 16 \
    --lr 3e-4 \
    --warmup_steps 915 \
    --max_steps 45776 \
    --use_fp8
```

### Resilient Training (Auto-Restart on Crash)

```bash
nohup bash train_1b_resilient.sh &
```

### Monitor Training

```bash
tail -F checkpoints/1b_final/nohup.out
```

---

## 📚 Training Data

| Metric | Value |
|--------|-------|
| **Total tokens** | ~41.1B (82 GB binary) |
| **Training subset** | ~21B tokens (Chinchilla-optimal for 1B model) |
| **Tokenizer** | Custom SentencePiece, 64K vocab |
| **Languages** | Korean, English, Code, Math |

### Data Sources

| Source | Domain |
|--------|--------|
| Cosmopedia | Web text, stories, textbooks |
| Korean C4 | Korean web crawl |
| Korean Wikipedia | Korean encyclopedia |
| NamuWiki | Korean wiki |
| CC-100 Korean | CommonCrawl Korean subset |
| MathPile | Mathematical text |
| OpenWebMath | Web-sourced math |
| HPLT Korean | High-performance language technologies |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| LR schedule | Cosine decay (min_lr_ratio=0.1) |
| Warmup steps | 915 |
| Total steps | 45,776 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Batch size | 16 per GPU (112 global) |
| Sequence length | 4,096 |
| Precision | FP8 (MXFP8BlockScaling) |
| Throughput | ~90,000 tok/s |
| Estimated time | ~64.8 hours |

---

## 🧬 Nemotron-Nano 아키텍처 단편화 도입 (Fragmented Adoption)

### 왜 "단편화 도입"인가?

NVIDIA의 Nemotron-H/Nano는 8B/4B 규모, 수천 GPU, 수조 토큰 학습을 전제로 설계된 아키텍처입니다. 이를 그대로 재현하는 것은 우리의 환경(7× B200, 65시간)에서 불가능합니다.

대신 **핵심 설계 원칙만을 추출(fragmentation)**하여, 제한된 하드웨어에 맞게 축소·적용했습니다. 이것이 "단편화 도입"의 의미입니다.

### Nemotron-Nano에서 가져온 것 vs 포기한 것

| Nemotron-Nano 원본 | 우리의 도입 방식 | 상태 |
|---|---|---|
| 대부분 Mamba-2, 소수 Attention (~9:1) | 16M + 2A (8:1 비율)로 유사하게 구성 | ✅ 도입 |
| Attention을 1/3, 2/3 지점에 배치 | 동일하게 등간격 배치 (18-layer: pos 6, 12) | ✅ 도입 |
| Mamba 블록 내부에 SwiGLU FFN 추가 | `mamba_d_ffn` config 필드로 구현 (0=비활성, 하위호환) | ✅ 도입 |
| Multi-head SSM with grouped heads | `mamba_n_groups=8`, `mamba_head_dim=64` | ✅ 도입 |
| GQA (Grouped Query Attention) | `n_kv_heads=4` (ratio 4:1) | ✅ 도입 |
| FP8 네이티브 학습 | TransformerEngine MXFP8BlockScaling | ✅ 도입 |
| 대규모 d_state (128) | `mamba_d_state=128` | ✅ 도입 |
| Chunked selective scan | `mamba_chunk_size=256` | ✅ 도입 |
| MoE (Mixture of Experts) | — | ❌ 포기 (소규모에서 효과 미미) |
| Knowledge Distillation | — | ❌ 포기 (teacher 모델 부재) |
| RLHF/DPO pipeline | — | ❌ 포기 (사전학습 단계) |
| 4B/8B 규모 | 994M으로 축소 | 🔄 스케일 조정 |
| 수조 토큰 학습 | 21B 토큰 (Chinchilla-optimal) | 🔄 스케일 조정 |

### 구체적 아키텍처 선정 과정

#### Step 1: 초기 3B 설계 (실패)

처음에는 Nemotron-Nano에 가까운 규모를 시도했습니다:

```
초기 설계: FRANKENSTALLM-H 3B
  d_model:     3072
  n_layers:    40 (37 Mamba-2 + 3 Attention)
  mamba_d_ffn: 4608
  n_groups:    8
  → 총 ~4.44B params
```

**문제 발견:** 65시간에 Chinchilla-optimal (20 × 4.44B = 88.8B tokens)의 **불과 7%**만 학습 가능. 심각한 undertrained 모델이 될 것이 확실했습니다. 이 규모에서는 약 930시간(39일)이 필요했습니다.

#### Step 2: 체계적 규모 탐색 (5개 모델 벤치마크)

Nemotron-H 스타일 아키텍처를 유지하면서 `d_model`, `n_layers`만 조정한 5개 config를 설계했습니다. 모든 config에서 다음 원칙을 유지:
- Mamba:Attention 비율 약 8~12:1
- Attention 레이어는 1/3, 2/3 지점
- `mamba_d_ffn = 1.5 × d_model`
- `mamba_n_groups = 8`, `mamba_head_dim = 64`

```
5개 후보 모델:
  1B:   d=2048, 18L (16M+2A)  →  994M params
  1.5B: d=2048, 28L (26M+2A)  → 1.48B params
  2B:   d=2560, 24L (22M+2A)  → 1.94B params
  2.5B: d=2560, 32L (30M+2A)  → 2.53B params
  3B:   d=3072, 26L (24M+2A)  → 2.95B params
```

각 모델을 7× B200에서 20 step 벤치마크하여 실측 throughput을 확인한 뒤, Chinchilla 달성률을 계산했습니다.

#### Step 3: 1B로 최종 결정

**Chinchilla Scaling Law** (Hoffmann et al., 2022): 동일 compute budget에서 "적정 크기 + 충분한 데이터"가 "큰 모델 + 부족한 데이터"를 항상 이깁니다.

```
1B:   90,455 tok/s × 65h = 21.2B tokens  →  Chinchilla 19.9B의 107%  ✅
1.5B: 59,107 tok/s × 65h = 13.8B tokens  →  Chinchilla 29.6B의  47%  ❌
2B:   51,076 tok/s × 65h = 11.9B tokens  →  Chinchilla 38.8B의  31%  ❌
```

1.5B는 필요 토큰의 절반만 학습하게 되어, 동일 크기의 fully-trained 모델보다 **오히려 성능이 떨어집니다**. 1B가 유일한 Chinchilla-optimal 후보였습니다.

#### 규모 축소의 의미

3B (4.44B params) → 1B (994M params)로의 축소는 단순한 타협이 아닙니다:

- **Fully-trained 1B > Under-trained 3B**: Chinchilla 법칙에 따르면, compute budget이 고정된 상황에서 작은 모델을 충분히 학습시키는 것이 큰 모델을 부족하게 학습시키는 것보다 모든 downstream task에서 우수
- **Nemotron-H 설계 원칙은 규모와 독립**: Mamba-Attention 하이브리드 패턴, SwiGLU FFN, GQA 등의 아키텍처 선택은 1B에서도 동일하게 유효
- **실험의 가치**: 소규모에서 아키텍처를 검증한 후, 더 큰 compute budget이 확보되면 동일 설계를 3B/7B로 스케일업 가능

---

## 📖 Development History

EVAFRILL-Mo는 6개 주요 단계를 거친 반복적 설계 여정의 결과물입니다.

### Phase 1 — FRANKENSTALLM (Pure Transformer)

순수 Transformer decoder-only LLM으로 시작. Custom SentencePiece 토크나이저를 Korean + English + Code + Math 데이터로 학습 (vocab 64,000). 기본 학습 파이프라인(DDP, checkpoint, cosine scheduler) 구축.

### Phase 2 — FRANKENSTALLM-H (Hybrid Evolution)

NVIDIA의 Nemotron-H 논문에서 영감을 받아 **하이브리드 Mamba-2 + Transformer** 아키텍처로 전환:

- Custom Mamba-2 selective scan 직접 구현 (`model/mamba_block.py`)
- Mamba 블록에 SwiGLU FFN 추가 (Nemotron-H의 핵심 혁신, `mamba_d_ffn` config)
- 초기 40-layer 3B 모델: 37 Mamba-2 + 3 Attention layers
- ~4.44B total parameters with `mamba_d_ffn=4608`
- 이름의 유래: **Frankenstein + POSCO STLLM + Hybrid**

### Phase 3 — 11-Step 구현 계획 (전체 완료)

1. **Config validation** — `LMConfig` dataclass with `__post_init__` divisibility checks
2. **Mamba FFN integration** — Optional SwiGLU, backward compatible (`mamba_d_ffn=0` 시 비활성)
3. **NaN detection** — 학습 중 NaN 감지 시 emergency checkpoint 저장
4. **CUDA kernel optimization** — Selective scan 성능 최적화
5. **Chunked Cross-Entropy** — logits 메모리 1/8 절감 (64K vocab에서 핵심)
6. **FP8 training** — TransformerEngine MXFP8BlockScaling on B200
7. **Gradient clipping & monitoring** — `max_grad_norm=1.0`, gnorm 추적
8. **Checkpoint save/resume** — Full DDP-compatible, optimizer/scheduler state 포함
9. **Cosine LR schedule** — Linear warmup + cosine decay (`min_lr_ratio=0.1`)
10. **Data pipeline optimization** — Memmap + `MADV_WILLNEED` + persistent workers
11. **Multi-GPU DDP** — 7× B200 분산 학습

### Phase 4 — Nemotron-Nano 단편화 도입 & 최적 규모 탐색 (EVAFRILL-Mo)

핵심 질문: **65시간 × 7 B200에서 Chinchilla-optimal 학습이 가능한 최대 모델 크기는?**

- Nemotron-Nano의 핵심 설계 원칙을 추출하여 5개 규모(1B~3B)에 적용 (상세: [단편화 도입 섹션](#-nemotron-nano-아키텍처-단편화-도입-fragmented-adoption))
- 5개 모델 체계적 벤치마크 (각 20 steps, 7 GPU)
- **Mamba Memory Cliff 현상 발견**: batch size 임계점에서 ~7.5× 메모리 점프
- **1B 모델 최종 선정**: 유일한 Chinchilla-optimal 후보 (107% 달성)

### Phase 5 — VectorDB / Memory DB 조사

LLM 사전학습에 vectorDB나 memoryDB가 도움이 되는지 조사:

| 접근법 | 조사 결과 | 판정 |
|--------|----------|------|
| RETRO-style 검색 증강 학습 | Mamba에 적용 불가 — CCA 레이어가 Transformer 전용 아키텍처 | ❌ 불가 |
| LMDB/RocksDB 데이터 로딩 | 2.2TB RAM에 82GB 데이터 전부 캐싱 → 개선 없음 | ❌ 불필요 |
| Curriculum Learning (DB 기반) | DB 없이도 가능, 1-3% 개선 수준 | ❌ DB 불필요 |
| FAISS/Milvus/LanceDB | 미설치 상태, 도입 오버헤드 과대 | ❌ 비용 초과 |

**결론:** 65시간 마감 하에서 구현 오버헤드가 학습 시간을 잠식하므로 도입 비추천. 순수 pretrain 집중이 최선.

### Phase 6 — Final 1B Training (현재 진행 중)

- **모델**: 994M params, 18 layers (16 Mamba-2 + 2 Attention)
- **학습**: 45,776 steps, ~21B tokens, ~64.8 hours
- **Throughput**: ~90,000 tok/s 안정적
- **Resilient wrapper**: `train_1b_resilient.sh` — 크래시 시 최신 체크포인트에서 자동 재시작 (최대 10회, 포트 자동 변경)

---

## 📊 Benchmark Results

### Model Size vs. Chinchilla Feasibility (65h Budget, 7× B200)

| Model | Params | Throughput (tok/s) | Max Batch | Mem/GPU | 65h Tokens | Chinchilla (20×) | Ratio |
|:------|-------:|---------:|----------:|--------:|----------:|----------------:|------:|
| **1B** | **994M** | **90,455** | **16** | **16.0 GB** | **21.2B** | **19.9B** | **107%** ✅ |
| 1.5B | 1.48B | 59,107 | 12 | 23.7 GB | 13.8B | 29.6B | 47% |
| 2B | 1.94B | 51,076 | 10 | 31.0 GB | 11.9B | 38.8B | 31% |
| 2.5B | 2.53B | 37,250 | 6 | 40.5 GB | 8.7B | 50.6B | 17% |
| 3B | 2.95B | 34,413 | 4 | 47.3 GB | 8.1B | 59.0B | 14% |

> Only the 1B model achieves **>100%** of the Chinchilla-optimal token budget within 65 hours.

### The Mamba Memory Cliff

A critical discovery during benchmarking: Mamba-2's selective scan exhibits a **dramatic memory cliff** at certain batch size thresholds.

- **Batch 12** → 23.7 GB/GPU
- **Batch 16** → 178 GB/GPU (7.5× increase)

This is caused by the selective scan materializing intermediate states when the product of batch size, sequence length, and state dimensions exceeds internal chunking boundaries. The key factors are `mamba_chunk_size=256` and `d_state=128`.

---

## 🙏 Acknowledgments

- **NVIDIA Nemotron-H** — For the hybrid Mamba-Transformer architecture design
- **Mamba-2** (Dao & Gu, 2024) — For the structured state space model foundation
- **Chinchilla scaling laws** (Hoffmann et al., 2022) — For optimal training compute allocation
- **Built with:** PyTorch, FlashAttention-2, TransformerEngine

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<div align="center">

*EVAFRILL-Mo — Built from scratch, one selective scan at a time.*

</div>
