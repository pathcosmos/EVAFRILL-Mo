<div align="center">

# EVAFRILL-Mo

**하이브리드 Mamba-2 + Transformer 언어 모델**

*Bride Eva (프랑켄슈타인의 신부) + FRIDAY (아이언맨 AI 비서) + LLM + Nemotron의 Mo*

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10.0--nv25.12-EE4C2C?logo=pytorch&logoColor=white)
![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![FlashAttention 2](https://img.shields.io/badge/FlashAttention-2.7.4-blueviolet)
![FP8](https://img.shields.io/badge/FP8-Native-orange)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![GPUs](https://img.shields.io/badge/GPUs-7%C3%97%20B200-76B900?logo=nvidia&logoColor=white)

NVIDIA [Nemotron-H](https://arxiv.org/abs/2504.03624) 아키텍처에서 영감을 받아 밑바닥부터 직접 구현한 **10억 파라미터 하이브리드 Mamba-2 + Transformer** 언어 모델입니다. 7× NVIDIA B200 GPU에서 Chinchilla-optimal 사전학습을 목표로 설계되었습니다.

</div>

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [아키텍처](#아키텍처)
- [Nemotron-Nano 단편화 도입](#nemotron-nano-아키텍처-단편화-도입)
- [하드웨어 환경](#하드웨어-환경)
- [프로젝트 구조](#프로젝트-구조)
- [빠른 시작](#빠른-시작)
- [학습 데이터](#학습-데이터)
- [개발 히스토리](#개발-히스토리)
- [벤치마크 결과](#벤치마크-결과)
- [감사의 글](#감사의-글)
- [라이선스](#라이선스)

---

## 프로젝트 개요

EVAFRILL-Mo는 **하이브리드 SSM-Transformer** 언어 모델을 밑바닥부터 직접 구현한 프로젝트입니다. 기존 모델 허브에 의존하지 않고, selective scan 커널부터 학습 루프까지 모든 구성 요소를 PyTorch로 직접 작성했습니다.

**주요 특징:**

- NVIDIA Nemotron-H 설계를 따른 **하이브리드 Mamba-2 + Transformer** 레이어 구성
- 커스텀 selective scan과 선택적 **SwiGLU FFN**을 갖춘 **Mamba-2 SSM**
- 효율적인 희소 어텐션 레이어를 위한 **GQA (Grouped Query Attention)**
- B200 GPU에서 **FP8 네이티브 학습** (MXFP8 블록 스케일링)
- logits 메모리 사용량을 1/8로 줄이는 **Chunked Cross-Entropy** 손실 함수
- **Chinchilla-optimal** 학습: 1B 모델을 ~21B 토큰으로 ~65시간 학습
- 한국어, 영어, 코드, 수학을 지원하는 커스텀 **SentencePiece 토크나이저** (64K 어휘)

---

## 아키텍처

### 1B 모델 구성

```
vocab_size:        64,000
d_model:           2,048
n_layers:          18  (Mamba-2 16개 + Attention 2개)
n_heads:           16
n_kv_heads:        4   (GQA 비율 4:1)
d_ffn:             5,504
mamba_d_ffn:       3,072  (Mamba 블록 내 SwiGLU FFN)
mamba_d_state:     128
mamba_head_dim:    64
mamba_n_groups:    8
mamba_chunk_size:  256
max_seq_len:       4,096
총 파라미터:        ~994M
```

### 하이브리드 레이어 배치

Mamba-2 SSM 블록 사이에 Transformer 어텐션 레이어를 네트워크의 약 1/3, 2/3 지점에 희소하게 배치합니다:

```
레이어 0-5:   Mamba-2 SSM  ──┐
레이어 6:     Attention (GQA) │  블록 1
레이어 7-11:  Mamba-2 SSM  ──┘
레이어 12:    Attention (GQA)    블록 2
레이어 13-17: Mamba-2 SSM
```

### 설계 원칙

| 구성 요소 | 설계 선택 | 근거 |
|-----------|----------|------|
| SSM 백본 | Mamba-2 selective scan | 선형 시간 시퀀스 모델링, 긴 문맥에서 효율적 |
| 희소 어텐션 | RoPE가 적용된 GQA | SSM이 놓칠 수 있는 전역 의존성 포착 |
| Mamba FFN | 선택적 SwiGLU | Nemotron-H의 혁신; scan 변경 없이 모델 용량 증가 |
| 손실 함수 | Chunked Cross-Entropy | logits를 청크 단위로 계산하여 최대 메모리 사용량 감소 |
| 정밀도 | FP8 (MXFP8BlockScaling) | B200 네이티브 지원, BF16 대비 ~2배 처리량 |
| 정규화 | RMSNorm | LayerNorm보다 빠르고 안정적 |

---

## Nemotron-Nano 아키텍처 단편화 도입

### "단편화 도입"이란?

NVIDIA의 Nemotron-H/Nano는 8B/4B 규모, 수천 GPU, 수조 토큰 학습을 전제로 설계된 아키텍처입니다. 이를 그대로 재현하는 것은 우리의 환경(7× B200, 65시간)에서 불가능합니다.

대신 **핵심 설계 원칙만을 추출(fragmentation)**하여, 제한된 하드웨어에 맞게 축소·적용했습니다. 이것이 "단편화 도입"의 의미입니다.

### 도입한 것 vs 포기한 것

| Nemotron-Nano 원본 | 우리의 도입 방식 | 상태 |
|---|---|---|
| 대부분 Mamba-2, 소수 Attention (~9:1) | 16M + 2A (8:1 비율)로 유사하게 구성 | ✅ 도입 |
| Attention을 1/3, 2/3 지점에 배치 | 동일하게 등간격 배치 (18-layer: 위치 6, 12) | ✅ 도입 |
| Mamba 블록 내부에 SwiGLU FFN 추가 | `mamba_d_ffn` config 필드로 구현 (0=비활성, 하위호환) | ✅ 도입 |
| Multi-head SSM with grouped heads | `mamba_n_groups=8`, `mamba_head_dim=64` | ✅ 도입 |
| GQA (Grouped Query Attention) | `n_kv_heads=4` (비율 4:1) | ✅ 도입 |
| FP8 네이티브 학습 | TransformerEngine MXFP8BlockScaling | ✅ 도입 |
| 대규모 d_state (128) | `mamba_d_state=128` | ✅ 도입 |
| 청크 기반 selective scan | `mamba_chunk_size=256` | ✅ 도입 |
| MoE (Mixture of Experts) | — | ❌ 포기 (소규모에서 효과 미미) |
| Knowledge Distillation | — | ❌ 포기 (teacher 모델 부재) |
| RLHF/DPO 파이프라인 | — | ❌ 포기 (사전학습 단계) |
| 4B/8B 규모 | 994M으로 축소 | 🔄 스케일 조정 |
| 수조 토큰 학습 | 21B 토큰 (Chinchilla-optimal) | 🔄 스케일 조정 |

### 구체적 아키텍처 선정 과정

#### 1단계: 초기 3B 설계 (실패)

처음에는 Nemotron-Nano에 가까운 규모를 시도했습니다:

```
초기 설계: FRANKENSTALLM-H 3B
  d_model:     3072
  n_layers:    40 (Mamba-2 37개 + Attention 3개)
  mamba_d_ffn: 4608
  n_groups:    8
  → 총 ~4.44B 파라미터
```

**발견된 문제:** 65시간에 Chinchilla-optimal (20 × 4.44B = 88.8B 토큰)의 **불과 7%**만 학습 가능. 심각한 미학습(undertrained) 모델이 될 것이 확실했습니다. 이 규모에서는 약 930시간(39일)이 필요했습니다.

#### 2단계: 체계적 규모 탐색 (5개 모델 벤치마크)

Nemotron-H 스타일 아키텍처를 유지하면서 `d_model`, `n_layers`만 조정한 5개 config를 설계했습니다. 모든 config에서 다음 원칙을 유지:
- Mamba:Attention 비율 약 8~12:1
- Attention 레이어는 1/3, 2/3 지점에 배치
- `mamba_d_ffn = 1.5 × d_model`
- `mamba_n_groups = 8`, `mamba_head_dim = 64`

```
5개 후보 모델:
  1B:   d=2048, 18L (16M+2A)  →  994M 파라미터
  1.5B: d=2048, 28L (26M+2A)  → 1.48B 파라미터
  2B:   d=2560, 24L (22M+2A)  → 1.94B 파라미터
  2.5B: d=2560, 32L (30M+2A)  → 2.53B 파라미터
  3B:   d=3072, 26L (24M+2A)  → 2.95B 파라미터
```

각 모델을 7× B200에서 20 step 벤치마크하여 실측 처리량을 확인한 뒤, Chinchilla 달성률을 계산했습니다.

#### 3단계: 1B로 최종 결정

**Chinchilla Scaling Law** (Hoffmann et al., 2022): 동일 compute budget에서 "적정 크기 + 충분한 데이터"가 "큰 모델 + 부족한 데이터"를 항상 이깁니다.

```
1B:   90,455 tok/s × 65h = 21.2B 토큰  →  Chinchilla 19.9B의 107%  ✅
1.5B: 59,107 tok/s × 65h = 13.8B 토큰  →  Chinchilla 29.6B의  47%  ❌
2B:   51,076 tok/s × 65h = 11.9B 토큰  →  Chinchilla 38.8B의  31%  ❌
```

1.5B는 필요 토큰의 절반만 학습하게 되어, 동일 크기의 완전 학습 모델보다 **오히려 성능이 떨어집니다**. 1B가 유일한 Chinchilla-optimal 후보였습니다.

#### 규모 축소의 의미

3B (4.44B 파라미터) → 1B (994M 파라미터)로의 축소는 단순한 타협이 아닙니다:

- **완전 학습된 1B > 미학습된 3B**: Chinchilla 법칙에 따르면, compute budget이 고정된 상황에서 작은 모델을 충분히 학습시키는 것이 큰 모델을 부족하게 학습시키는 것보다 모든 다운스트림 태스크에서 우수
- **Nemotron-H 설계 원칙은 규모와 독립**: Mamba-Attention 하이브리드 패턴, SwiGLU FFN, GQA 등의 아키텍처 선택은 1B에서도 동일하게 유효
- **실험의 가치**: 소규모에서 아키텍처를 검증한 후, 더 큰 compute budget이 확보되면 동일 설계를 3B/7B로 스케일업 가능

---

## 하드웨어 환경

| 항목 | 사양 |
|------|------|
| **GPU** | 7× NVIDIA B200 (GPU당 183 GB VRAM, 총 ~1.28 TB) |
| **시스템 RAM** | 2.2 TB |
| **CUDA** | 13.0 |
| **스토리지** | GPFS 20 TB (여유 9 TB) |
| **PyTorch** | 2.10.0a0+nv25.12 (NVIDIA 커스텀 빌드, B200 최적화) |
| **FlashAttention** | 2.7.4.post1+25.12 |

> **주의:** PyTorch는 NVIDIA 커스텀 빌드(`nv25.12`)입니다. `pip install torch`로 재설치하면 B200 최적화가 깨지므로 **절대 재설치하지 마세요**.

---

## 프로젝트 구조

```
EVAFRILL-Mo/
├── README.md                  # 이 파일
├── CLAUDE.md                  # AI 어시스턴트 지시사항
│
├── model/                     # 모델 아키텍처
│   ├── config.py              # LMConfig 데이터클래스 (__post_init__ 검증 포함)
│   ├── transformer.py         # LLM 메인 모델 (하이브리드 레이어 디스패처)
│   ├── mamba_block.py         # Mamba-2 SSM + 선택적 SwiGLU FFN
│   ├── attention.py           # RoPE가 적용된 GQA 어텐션
│   └── layers.py              # RMSNorm, SwiGLU, 임베딩
│
├── train/                     # 학습
│   ├── pretrain.py            # 사전학습 엔트리포인트
│   ├── trainer.py             # 학습 루프 (DDP, FP8, 체크포인팅)
│   ├── sft.py                 # 지도 미세조정 (SFT)
│   ├── orpo.py                # ORPO 선호도 최적화
│   └── utils.py               # Cosine 스케줄러, DDP 설정, 체크포인트 유틸
│
├── data/                      # 데이터 파이프라인
│   ├── dataset.py             # PackedDataset (memmap + MADV_WILLNEED 힌트)
│   ├── prepare.py             # 토큰화 파이프라인
│   └── *.bin                  # 바이너리 토큰 파일 (저장소에 미포함)
│
├── eval/                      # 평가
│   ├── perplexity.py          # 퍼플렉시티 평가
│   ├── generate.py            # 텍스트 생성 / 샘플링
│   └── comprehensive_eval.py  # 종합 평가 도구
│
├── configs/                   # YAML 학습 설정 파일
├── scripts/                   # 실행, 모니터링, 배포 스크립트
├── benchmarks/                # 처리량 & 프로파일링 도구
├── tokenizer/                 # SentencePiece 토크나이저 학습
├── reports/                   # 평가 및 분석 리포트
├── docs/                      # 하드웨어 & 환경 문서
└── train_1b_resilient.sh      # 자동 재시작 학습 래퍼 (크래시 복구)
```

---

## 빠른 시작

### 사전 요구 사항

```bash
# 필요 라이브러리 설치 (PyTorch는 사전 설치됨 — 재설치 금지)
pip install transformers accelerate peft trl deepspeed bitsandbytes sentencepiece wandb
```

### 단일 GPU 테스트

```bash
python train/pretrain.py \
    --config configs/small.yaml \
    --train_data data/train.bin \
    --batch_size 8
```

### 멀티 GPU 학습 (7× B200, FP8)

```bash
torchrun --nproc_per_node=7 train/pretrain.py \
    --config configs/1b_hybrid.yaml \
    --train_data data/3b_train.bin \
    --batch_size 16 \
    --lr 3e-4 \
    --warmup_steps 915 \
    --max_steps 45776 \
    --use_fp8
```

### 자동 재시작 학습 (크래시 시 자동 복구)

```bash
nohup bash train_1b_resilient.sh &
```

### 학습 모니터링

```bash
tail -F checkpoints/1b_final/nohup.out
```

---

## 적용 기술 상세

이 프로젝트에 적용된 핵심 기술들을 빠짐없이 정리합니다.

### SSM / Mamba-2 관련

| 기술 | 설명 | 적용 위치 |
|------|------|-----------|
| **Triton Chunked SSD 커널** | `mamba_ssm`의 `mamba_chunk_scan_combined` — Triton으로 작성된 chunked Structured State Space Duality 커널. 메모리 효율적인 O(N) 시퀀스 처리 | `model/mamba_block.py:333` |
| **causal_conv1d** | 퓨즈드 CUDA 커널로 causal depthwise conv1d + SiLU 활성화를 단일 커널에서 처리 | `model/mamba_block.py:312` |
| **Selective Scan (순수 PyTorch 폴백)** | CUDA 커널 미설치 시를 위한 순수 PyTorch selective scan 구현. 청크 기반으로 메모리 효율성 확보 | `model/mamba_block.py:54` |
| **Multi-head SSM** | 64개 헤드를 8개 그룹으로 나눈 grouped SSM. Mamba-2의 핵심 구조 | `mamba_n_groups=8`, `mamba_head_dim=64` |
| **A_log 파라미터화** | 대각 감쇠 행렬 A를 log 공간에서 학습하여 수치 안정성 보장. `exp(-exp(A_log) * dt)` | `model/mamba_block.py:219` |
| **dt_bias 초기화** | 시간 스텝 바이어스를 `log(uniform(0.001, 0.1))`로 초기화하여 학습 초기 안정성 확보 | `model/mamba_block.py:227` |
| **Mamba SwiGLU FFN** | Nemotron-H 스타일로 Mamba 블록 내부에 SwiGLU FFN 추가. `mamba_d_ffn=0`이면 비활성 (하위 호환) | `model/mamba_block.py` |

### Transformer / Attention 관련

| 기술 | 설명 | 적용 위치 |
|------|------|-----------|
| **FlashAttention-2** | Tri Dao의 IO-aware 어텐션 알고리즘. O(N)메모리로 정확한 어텐션 계산 | `model/attention.py:211` |
| **GQA (Grouped Query Attention)** | 16개 쿼리 헤드, 4개 KV 헤드 (4:1 비율). KV 캐시 메모리 75% 절감 | `model/attention.py:77` |
| **RoPE (Rotary Positional Embedding)** | 회전 위치 임베딩으로 상대적 위치 정보 인코딩. `rope_theta=500000` | `model/layers.py:54`, `model/attention.py:39` |
| **RMSNorm** | LayerNorm 대비 연산량 감소 (mean 계산 불필요). Pre-norm 구조 | `model/layers.py:27` |
| **SwiGLU FFN** | Shazeer(2020)의 SwiGLU 게이트 활성화. `gate * silu(up)` 구조 | `model/layers.py:109` |

### 정밀도 / 양자화

| 기술 | 설명 | 적용 위치 |
|------|------|-----------|
| **FP8 (MXFP8BlockScaling)** | TransformerEngine의 Microscaling FP8. B200의 FP8 텐서 코어를 활용하여 BF16 대비 ~2배 처리량 | `train/trainer.py:163` |
| **fp8_autocast** | TE 모듈(te.Linear)만 FP8로 연산, 나머지는 BF16 유지하는 하이브리드 정밀도 | `train/trainer.py:470` |
| **BF16 autocast** | `torch.autocast(dtype=bfloat16)` — 순수 PyTorch 레이어(Mamba)는 BF16으로 자동 캐스팅 | `train/trainer.py:467` |
| **te.Linear (FP8 Linear)** | Attention 레이어의 QKV/Output 프로젝션에 TransformerEngine FP8 Linear 적용 | `model/attention.py:103` |
| **FP8 정렬 검증** | `d_model`, `d_ffn`, `mamba_d_ffn` 모두 16의 배수인지 `__post_init__`에서 검증 | `model/config.py:120` |

### 손실 함수 / 메모리 최적화

| 기술 | 설명 | 적용 위치 |
|------|------|-----------|
| **Chunked Cross-Entropy** | 전체 logits (B×T×V)를 한번에 계산하지 않고 청크 단위로 분할. 64K 어휘에서 logits 메모리 1/8로 절감 | `model/transformer.py:232` |
| **Gradient Accumulation + no_sync** | DDP에서 accumulation step 동안 `model.no_sync()`로 불필요한 allreduce 방지 | `train/trainer.py:243` |
| **gradient_as_bucket_view** | DDP의 gradient 버퍼를 NCCL 통신 버킷으로 직접 사용. 메모리 복사 제거 (zero-copy) | `train/pretrain.py:323` |

### 분산 학습 / 하드웨어 최적화

| 기술 | 설명 | 적용 위치 |
|------|------|-----------|
| **DDP (DistributedDataParallel)** | 7× B200 GPU 간 데이터 병렬 학습. NCCL 백엔드 | `train/pretrain.py:317` |
| **NUMA 어피니티** | GPU 0-3 → NUMA 노드 0 (코어 0-35), GPU 4-6 → NUMA 노드 1 (코어 36-71). 메모리 접근 지연 3.2배 감소 | `train/pretrain.py:256` |
| **DistributedSampler** | 데이터를 GPU 간 균등 분배하여 중복 학습 방지 | `train/pretrain.py:335` |
| **expandable_segments** | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — CUDA 메모리 단편화 방지 | 환경 변수 |

### 데이터 파이프라인

| 기술 | 설명 | 적용 위치 |
|------|------|-----------|
| **np.memmap** | 학습 데이터를 메모리 매핑하여 디스크에서 직접 읽기. 82GB 데이터를 RAM에 전부 매핑 | `data/dataset.py:38` |
| **MADV_RANDOM** | 랜덤 액세스 패턴을 커널에 알려 불필요한 read-ahead 비활성화 | `data/dataset.py:95` |
| **MADV_WILLNEED** | 비동기적으로 페이지를 페이지 캐시에 프리폴트 (prefault) | `data/dataset.py:96` |
| **persistent_workers** | DataLoader 워커를 에포크 간 유지. 워커 재생성 오버헤드 제거 | `train/pretrain.py:355` |
| **pin_memory** | CPU→GPU 전송을 위한 페이지 고정 메모리. DMA 전송 가속 | `train/pretrain.py:352` |
| **prefetch_factor=4** | 워커당 4배치를 미리 로드하여 GPU 대기 시간 최소화 | `train/pretrain.py:354` |
| **6 워커/GPU** | 6×7=42 워커, 72코어 CPU 예산 내에서 OMP_NUM_THREADS=4와 균형 | `train/pretrain.py:351` |

### 학습 안정성 / 스케줄링

| 기술 | 설명 | 적용 위치 |
|------|------|-----------|
| **Cosine LR Schedule + 선형 워밍업** | 워밍업 후 cosine 감쇠로 학습률 조절. `min_lr_ratio=0.1` (최종 lr = 3e-5) | `train/utils.py:35` |
| **AdamW (weight_decay 선택적 적용)** | bias, RMSNorm, A_log, D, dt_bias 파라미터는 weight decay에서 제외 | `train/pretrain.py:203` |
| **Gradient Clipping (max_norm=1.0)** | L2 norm 기반 기울기 클리핑. Mamba의 기울기 스파이크 방지 | `train/trainer.py:280` |
| **NaN 감지 + 긴급 체크포인트** | 학습 중 NaN/Inf 감지 시 즉시 체크포인트 저장 후 경고 | `model/mamba_block.py:349` |
| **자동 재시작 래퍼** | 크래시 시 최신 체크포인트에서 자동 재시작. 포트 자동 변경 (EADDRINUSE 방지) | `train_1b_resilient.sh` |

### 토크나이저

| 기술 | 설명 | 적용 위치 |
|------|------|-----------|
| **SentencePiece BPE** | 64K 어휘의 Byte-Pair Encoding. 한국어+영어+코드+수학 혼합 학습 | `tokenizer/` |
| **HuggingFace 호환 변환** | SentencePiece 모델을 HF tokenizer 형식으로 변환 | `tokenizer/convert_sp_to_hf.py` |

---

## 학습 데이터

| 항목 | 값 |
|------|-----|
| **총 토큰 수** | ~41.1B (82 GB 바이너리) |
| **학습 사용량** | ~21B 토큰 (1B 모델 Chinchilla-optimal) |
| **토크나이저** | 커스텀 SentencePiece, 64K 어휘 |
| **지원 언어** | 한국어, 영어, 코드, 수학 |

### 데이터 소스

| 소스 | 도메인 |
|------|--------|
| Cosmopedia | 웹 텍스트, 이야기, 교과서 |
| Korean C4 | 한국어 웹 크롤 |
| 한국어 위키백과 | 한국어 백과사전 |
| 나무위키 | 한국어 위키 |
| CC-100 한국어 | CommonCrawl 한국어 부분집합 |
| MathPile | 수학 텍스트 |
| OpenWebMath | 웹 기반 수학 데이터 |
| HPLT 한국어 | 고성능 언어 기술 데이터 |

### 학습 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| 학습률 | 3e-4 |
| 학습률 스케줄 | Cosine 감쇠 (min_lr_ratio=0.1) |
| 워밍업 스텝 | 915 |
| 총 스텝 | 45,776 |
| 가중치 감쇠 | 0.1 |
| 기울기 클리핑 | 1.0 |
| 배치 크기 | GPU당 16 (전체 112) |
| 시퀀스 길이 | 4,096 |
| 정밀도 | FP8 (MXFP8BlockScaling) |
| 처리량 | ~90,000 tok/s |
| 예상 소요 시간 | ~64.8시간 |

---

## 개발 히스토리

EVAFRILL-Mo는 6개 주요 단계를 거친 반복적 설계 여정의 결과물입니다.

### 1단계 — FRANKENSTALLM (순수 Transformer)

순수 Transformer decoder-only LLM으로 시작했습니다. 한국어 + 영어 + 코드 + 수학 데이터로 커스텀 SentencePiece 토크나이저를 학습시켰으며 (어휘 64,000), 기본 학습 파이프라인(DDP, 체크포인트, cosine 스케줄러)을 구축했습니다.

### 2단계 — FRANKENSTALLM-H (하이브리드 진화)

NVIDIA의 Nemotron-H 논문에서 영감을 받아 **하이브리드 Mamba-2 + Transformer** 아키텍처로 전환했습니다:

- Mamba-2 selective scan 직접 구현 (`model/mamba_block.py`)
- Mamba 블록에 SwiGLU FFN 추가 (Nemotron-H의 핵심 혁신, `mamba_d_ffn` config)
- 초기 40층 3B 모델: Mamba-2 37개 + Attention 3개 레이어
- `mamba_d_ffn=4608`로 총 ~4.44B 파라미터
- 이름의 유래: **Frankenstein + POSCO STLLM + Hybrid**

### 3단계 — 11단계 구현 계획 (전체 완료)

1. **Config 검증** — `LMConfig` 데이터클래스의 `__post_init__` 나눗셈 검사
2. **Mamba FFN 통합** — 선택적 SwiGLU, 하위 호환 (`mamba_d_ffn=0`이면 비활성)
3. **NaN 감지** — 학습 중 NaN 감지 시 긴급 체크포인트 저장
4. **CUDA 커널 최적화** — Selective scan 성능 최적화
5. **Chunked Cross-Entropy** — logits 메모리 1/8 절감 (64K 어휘에서 핵심)
6. **FP8 학습** — B200에서 TransformerEngine MXFP8BlockScaling
7. **기울기 클리핑 & 모니터링** — `max_grad_norm=1.0`, gnorm 추적
8. **체크포인트 저장/복원** — 완전한 DDP 호환, optimizer/scheduler 상태 포함
9. **Cosine 학습률 스케줄** — 선형 워밍업 + cosine 감쇠 (`min_lr_ratio=0.1`)
10. **데이터 파이프라인 최적화** — Memmap + `MADV_WILLNEED` + persistent workers
11. **멀티 GPU DDP** — 7× B200 분산 학습

### 4단계 — Nemotron-Nano 단편화 도입 & 최적 규모 탐색 (EVAFRILL-Mo)

핵심 질문: **65시간 × 7 B200에서 Chinchilla-optimal 학습이 가능한 최대 모델 크기는?**

- Nemotron-Nano의 핵심 설계 원칙을 추출하여 5개 규모(1B~3B)에 적용 (상세: [단편화 도입 섹션](#nemotron-nano-아키텍처-단편화-도입))
- 5개 모델 체계적 벤치마크 (각 20 steps, 7 GPU)
- **Mamba Memory Cliff 현상 발견**: 배치 크기 임계점에서 ~7.5배 메모리 점프
- **1B 모델 최종 선정**: 유일한 Chinchilla-optimal 후보 (107% 달성)

### 5단계 — VectorDB / Memory DB 조사

LLM 사전학습에 vectorDB나 memoryDB가 도움이 되는지 조사했습니다:

| 접근법 | 조사 결과 | 판정 |
|--------|----------|------|
| RETRO 스타일 검색 증강 학습 | Mamba에 적용 불가 — CCA 레이어가 Transformer 전용 아키텍처 | ❌ 불가 |
| LMDB/RocksDB 데이터 로딩 | 2.2TB RAM에 82GB 데이터 전부 캐싱됨 → 개선 없음 | ❌ 불필요 |
| Curriculum Learning (DB 기반) | DB 없이도 가능, 1-3% 개선 수준 | ❌ DB 불필요 |
| FAISS/Milvus/LanceDB | 미설치 상태, 도입 오버헤드 과대 | ❌ 비용 초과 |

**결론:** 65시간 마감 하에서 구현 오버헤드가 학습 시간을 잠식하므로 도입 비추천. 순수 사전학습에 집중하는 것이 최선.

### 6단계 — 최종 1B 학습 (현재 진행 중)

- **모델**: 994M 파라미터, 18층 (Mamba-2 16개 + Attention 2개)
- **학습**: 45,776 스텝, ~21B 토큰, ~64.8시간
- **처리량**: ~90,000 tok/s 안정적
- **복구 래퍼**: `train_1b_resilient.sh` — 크래시 시 최신 체크포인트에서 자동 재시작 (최대 10회, 포트 자동 변경)

---

## 벤치마크 결과

### 모델 규모별 Chinchilla 달성 가능성 (65시간, 7× B200)

| 모델 | 파라미터 | 처리량 (tok/s) | 최대 배치 | GPU당 메모리 | 65h 토큰 | Chinchilla (20×) | 달성률 |
|:-----|--------:|---------:|----------:|--------:|----------:|----------------:|------:|
| **1B** | **994M** | **90,455** | **16** | **16.0 GB** | **21.2B** | **19.9B** | **107%** ✅ |
| 1.5B | 1.48B | 59,107 | 12 | 23.7 GB | 13.8B | 29.6B | 47% |
| 2B | 1.94B | 51,076 | 10 | 31.0 GB | 11.9B | 38.8B | 31% |
| 2.5B | 2.53B | 37,250 | 6 | 40.5 GB | 8.7B | 50.6B | 17% |
| 3B | 2.95B | 34,413 | 4 | 47.3 GB | 8.1B | 59.0B | 14% |

> 65시간 내에 Chinchilla-optimal 토큰 예산의 **100% 이상**을 달성하는 모델은 1B뿐입니다.

### Mamba Memory Cliff 현상

벤치마크 중 발견한 중요한 현상: Mamba-2의 selective scan은 특정 배치 크기 임계점에서 **극적인 메모리 절벽(cliff)**을 보입니다.

```
1.5B 모델 기준:
  배치 12 → 23.7 GB/GPU
  배치 16 → 178  GB/GPU (7.5배 증가!)
```

이는 배치 크기, 시퀀스 길이, 상태 차원의 곱이 내부 청킹 경계를 초과할 때 selective scan이 중간 상태를 완전히 메모리에 구체화(materialize)하기 때문입니다. 핵심 요인은 `mamba_chunk_size=256`과 `d_state=128`입니다.

---

## 감사의 글

- **NVIDIA Nemotron-H** — 하이브리드 Mamba-Transformer 아키텍처 설계의 영감
- **Mamba-2** (Dao & Gu, 2024) — 구조화된 상태 공간 모델의 기반
- **Chinchilla 스케일링 법칙** (Hoffmann et al., 2022) — 최적 학습 compute 배분 기준
- **사용 기술:** PyTorch, FlashAttention-2, TransformerEngine

---

## 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다. 상세 내용은 [LICENSE](LICENSE)를 참조하세요.

---

<div align="center">

*EVAFRILL-Mo — 밑바닥부터, selective scan 하나하나 직접 쌓아올린 모델.*

</div>
