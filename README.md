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

NVIDIA [Nemotron-H](https://arxiv.org/abs/2504.03624) 아키텍처에서 영감을 받아 밑바닥부터 직접 구현한 **30억 파라미터 하이브리드 Mamba-2 + Transformer** 언어 모델입니다. 7× NVIDIA B200 GPU에서 60시간 Chinchilla-optimal 사전학습을 목표로 설계되었습니다.

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
- [1B → 3B 전환 경위](#1b--3b-전환-경위)
- [3B 하드웨어 제약 최적화](#3b-하드웨어-제약-최적화)
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
- **Chinchilla-optimal** 학습: 3B 모델을 ~55B 토큰으로 ~60시간 학습
- 한국어, 영어, 코드, 수학을 지원하는 커스텀 **SentencePiece 토크나이저** (64K 어휘)

---

## 아키텍처

### 3B 모델 구성 (현재 학습 중)

```
vocab_size:        64,000
d_model:           3,072
n_layers:          26  (Mamba-2 24개 + Attention 2개)
n_heads:           24
n_kv_heads:        8   (GQA 비율 3:1)
d_ffn:             9,216
mamba_d_ffn:       4,608  (Mamba 블록 내 SwiGLU FFN)
mamba_d_state:     128
mamba_head_dim:    64
mamba_n_groups:    8
mamba_chunk_size:  256
max_seq_len:       4,096
총 파라미터:        ~2,944M (2.94B)
```

### 이전 1B 모델 구성 (실험 완료)

```
d_model: 2,048 | n_layers: 18 (16M+2A) | n_heads: 16 | n_kv_heads: 4
d_ffn: 5,504 | mamba_d_ffn: 3,072 | 총 파라미터: ~994M
```

### 하이브리드 레이어 배치

Mamba-2 SSM 블록 사이에 Transformer 어텐션 레이어를 네트워크의 약 1/2 지점과 마지막에 희소하게 배치합니다:

```
3B 레이어 배치 (26층):
레이어 0-11:  Mamba-2 SSM ×12  ──┐
레이어 12:    Attention (GQA)     │  전반부
레이어 13-24: Mamba-2 SSM ×12  ──┘
레이어 25:    Attention (GQA)        후반부
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
├── train_1b_resilient.sh      # 1B 자동 재시작 학습 래퍼
└── train_3b_resilient.sh      # 3B 자동 재시작 학습 래퍼 (현재 사용 중)
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

### 멀티 GPU 학습 — 3B 모델 (7× B200, FP8)

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

### 자동 재시작 학습 (크래시 시 자동 복구)

```bash
nohup bash train_3b_resilient.sh &
```

### 학습 모니터링

```bash
# 학습 로그 (step별 loss, tok/s, lr)
tail -F checkpoints/3b_final/train.log

# 재시작/에러 이벤트 모니터
tail -F checkpoints/3b_final/monitor.log
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

## 1B → 3B 전환 경위

### 발견: tok/s는 per-GPU였다

1B 모델 학습을 시작한 후, 예상보다 훨씬 빠르게 진행되는 것을 감지했습니다.

```
1B 학습 시작 후 ~1시간:
  step 3,700 / 45,776 (8.1%)
  경과 시간: 0.8시간
  예상 완료: ~9.3시간
```

**원인: 처리량 지표의 오해석.** `trainer.py`의 `tokens_per_sec` 계산은 로컬(per-GPU) 값이었습니다:

```python
# trainer.py:335 — batch_size는 로컬(per-GPU) 배치
tokens_per_sec = (batch_size * seq_len * grad_accum * log_interval) / elapsed
```

즉 로그의 `tok/s 90,000`은 **GPU 1개의 처리량**이었고, 실제 전체 처리량은:

```
실제 aggregate: 90,000 × 7 GPU = 630,000 tok/s
```

### 재계산: 1B는 65시간의 1/7만 필요

| 항목 | 이전 계산 (잘못됨) | 수정된 계산 |
|------|:--:|:--:|
| tok/s | 90,000 (aggregate) | 630,000 (aggregate) |
| 65h 토큰 | 21.1B | **147.4B** |
| Chinchilla 달성 | 107% | **751%** |
| 실제 필요 시간 | ~64.8h | **~8.8h** |

**1B 모델에 65시간을 투자하면 Chinchilla의 7.5배를 학습하는 극심한 과잉학습(over-training)**이 됩니다. Compute budget이 크게 남는다는 의미이므로, 훨씬 큰 모델을 학습할 수 있습니다.

### 3B 전환 결정

수정된 계산으로 전체 모델 규모를 재평가했습니다:

| 모델 | tok/s (agg) | 60h 토큰 | Chinchilla | 달성률 |
|------|----------:|--------:|---------:|------:|
| 1B | 630,000 | 136.1B | 20B | 681% (과잉) |
| 1.5B | 367,213 | 79.3B | 30B | 264% (과잉) |
| 2B | 271,894 | 58.7B | 38B | 155% (과잉) |
| 2.5B | 260,519 | 56.3B | 50B | 113% |
| **3B** | **254,681** | **55.0B** | **58.9B** | **93%** |

**3B가 60시간 예산에서 Chinchilla의 93%를 달성할 수 있는 최대 규모 모델**입니다. 진행 중이던 1B 학습(step 4,230)을 중단하고 3B로 전환했습니다.

---

## 3B 하드웨어 제약 최적화

### 핵심 제약: Mamba Memory Cliff

3B 벤치마크에서 **배치 크기 6→7에서 OOM이 발생**했습니다. 이는 Mamba-2의 Triton Chunked SSD 커널이 특정 임계점에서 중간 텐서(intermediate states)를 완전히 구체화(materialize)하기 때문입니다.

```
3B 모델 배치 크기 테스트 결과 (7× B200, FP8):
  batch=6  →  47.3 GB/GPU  ✅ (안정)
  batch=7  →  OOM          ❌ (Memory Cliff)
  batch=8  →  OOM          ❌
  batch=10 →  OOM          ❌
  batch=12 →  OOM          ❌
```

**Cliff 발생 메커니즘:** `mamba_chunk_scan_combined` 커널은 `(batch, n_chunks, n_heads, chunk_size, d_state)` 크기의 중간 텐서를 할당합니다. batch=6까지는 이를 청크별로 스트리밍하지만, batch=7부터는 전체를 메모리에 구체화하여 **47GB → 183GB+** 로 폭증합니다.

### 최적화된 3B 학습 설정

Cliff 이하의 최대 배치(batch=6)에서 처리량을 극대화하는 설정입니다:

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| **batch_size** | 6 (per-GPU) | Memory Cliff 직전 최대값. 47.3GB/183GB |
| **grad_accum** | 1 | 추가 accumulation은 처리량 증가 없음 (wall clock 동일) |
| **effective_batch** | 42 seqs (172,032 tok) | 6 × 7 GPU × 4,096 seq_len |
| **lr** | 3e-4 | 3B 규모 표준 학습률 |
| **warmup_steps** | 6,395 | 총 steps의 2% (과도한 초기 그래디언트 방지) |
| **max_steps** | 319,772 | 55B tokens / 172,032 tok/step |
| **weight_decay** | 0.1 | AdamW 표준 (bias, norm, SSM 파라미터 제외) |
| **정밀도** | FP8 (MXFP8BlockScaling) | BF16 대비 ~2배 처리량 |
| **max_grad_norm** | 1.0 | Mamba 그래디언트 스파이크 방지 |
| **min_lr_ratio** | 0.1 | 최종 lr = 3e-5 |
| **seed** | 42 | 재현성 보장 |

### 처리량 분석

```
3B 모델 실측 성능:
  per-GPU:   36,383 tok/s
  aggregate:  254,681 tok/s (×7 GPUs)
  step time:  ~0.67s/step
  GPU 메모리:  47.3 GB / 183 GB (25.8% 사용)
  GPU 활용:   거의 100% (compute-bound)
```

### 메모리 효율 분석

batch=6에서 GPU 메모리의 25.8%만 사용하지만, Mamba Memory Cliff로 인해 batch=7부터는 183GB를 초과합니다. 이 74.2%의 "남는" VRAM은 Mamba SSM의 구조적 제약으로 인해 활용이 불가합니다.

```
메모리 분해 (추정):
  모델 가중치 (FP8):    ~3.0 GB
  옵티마이저 상태:       ~18.0 GB (AdamW, FP32 moments)
  기울기 버퍼:          ~6.0 GB
  활성화 (batch=6):     ~20.3 GB
  ──────────────────────────────
  합계:                 ~47.3 GB
```

### 데이터 제약

| 항목 | 값 |
|------|-----|
| 학습 데이터 | 41.1B 토큰 (82 GB) |
| 60h 처리 가능 | 55.0B 토큰 |
| 필요 에포크 | ~1.34 |
| Chinchilla 달성 | ~93% (1 epoch: 70%, 1.34 epoch: 93%) |

1.34 에포크의 데이터 반복은 수용 가능합니다 — Chinchilla 논문 자체도 1-2 에포크 범위의 데이터 반복을 허용하며, 최근 연구(Muennighoff et al., 2023)에 따르면 최대 4 에포크까지 성능 저하가 미미합니다.

### 자동 복구 시스템

60시간 연속 학습 안정성을 위한 `train_3b_resilient.sh`:

```
복구 매커니즘:
  1. 크래시 감지 (exit code ≠ 0)
  2. GPU 프로세스 강제 종료 + 메모리 해제 대기
  3. 최신 체크포인트 자동 탐색 (checkpoint-XXXXXXX)
  4. 포트 번호 자동 증가 (EADDRINUSE 방지)
  5. 30초 대기 후 재시작
  6. 최대 10회 재시도
```

---

## 학습 데이터

| 항목 | 값 |
|------|-----|
| **총 토큰 수** | ~41.1B (82 GB 바이너리) |
| **학습 사용량** | ~55B 토큰 (3B 모델, ~1.34 에포크) |
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

### 학습 하이퍼파라미터 (3B 본학습)

| 파라미터 | 값 |
|---------|-----|
| 학습률 | 3e-4 |
| 학습률 스케줄 | Cosine 감쇠 (min_lr_ratio=0.1) |
| 워밍업 스텝 | 6,395 (총 steps의 2%) |
| 총 스텝 | 319,772 |
| 가중치 감쇠 | 0.1 |
| 기울기 클리핑 | 1.0 |
| 배치 크기 | GPU당 6 (전체 42) — Memory Cliff 제약 |
| 시퀀스 길이 | 4,096 |
| 정밀도 | FP8 (MXFP8BlockScaling) |
| 처리량 | ~36,383 tok/s (per-GPU), ~254,681 tok/s (aggregate) |
| 예상 소요 시간 | ~60시간 |
| Chinchilla 달성률 | ~93% |

### 이전 1B 학습 하이퍼파라미터 (실험용)

| 파라미터 | 값 |
|---------|-----|
| 배치 크기 | GPU당 16 (전체 112) |
| 총 스텝 | 45,776 |
| 처리량 | ~90,000 tok/s (per-GPU), ~630,000 tok/s (aggregate) |
| 실제 소요 시간 | ~8.8시간 (중단됨, step 4,230에서 3B 전환) |

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

### 6단계 — 1B 학습 시작 & 과잉학습 감지

- **모델**: 994M 파라미터, 18층 (Mamba-2 16개 + Attention 2개)
- **학습 시작**: 45,776 스텝, batch=16, ~90,000 tok/s (per-GPU)
- **감지**: step 3,700 시점에서 전체 소요 시간이 ~9.3시간으로 예측됨
- **원인 분석**: tok/s가 per-GPU 값임을 확인 → 실제 aggregate는 630,000 tok/s
- **판단**: 1B에 65시간 투자 시 Chinchilla의 7.5배 과잉학습 → compute 낭비
- **결정**: step 4,230에서 1B 학습 중단, 3B 규모로 전환

### 7단계 — 3B 최적화 & 본학습 (현재 진행 중)

- **모델**: 2,944M 파라미터, 26층 (Mamba-2 24개 + Attention 2개)
- **벤치마크**: batch=6~12까지 순차 테스트 → batch=6이 Memory Cliff 직전 최대값
- **처리량**: 36,383 tok/s (per-GPU), 254,681 tok/s (aggregate)
- **학습**: 319,772 스텝, ~55B 토큰, ~60시간
- **Chinchilla 달성률**: ~93% (1.34 에포크)
- **복구 래퍼**: `train_3b_resilient.sh` — 크래시 시 최신 체크포인트에서 자동 재시작 (최대 10회, 포트 자동 변경)

---

## 벤치마크 결과

### 모델 규모별 Chinchilla 달성 가능성 (60시간, 7× B200)

> **주의:** tok/s는 **per-GPU** 값입니다. 전체(aggregate) 처리량은 ×7입니다.

| 모델 | 파라미터 | tok/s (per-GPU) | tok/s (agg ×7) | 최대 배치 | GPU당 메모리 | 60h 토큰 | Chinchilla | 달성률 |
|:-----|--------:|----------:|----------:|-----:|--------:|---------:|----------:|------:|
| 1B | 994M | 90,000 | 630,000 | 16 | 16.0 GB | 136.1B | 19.9B | 681% |
| 1.5B | 1.48B | 52,459 | 367,213 | 12 | 23.7 GB | 79.3B | 29.6B | 268% |
| 2B | 1.94B | 38,842 | 271,894 | 10 | 31.0 GB | 58.7B | 38.8B | 151% |
| 2.5B | 2.53B | 37,217 | 260,519 | 6 | 40.5 GB | 56.3B | 50.6B | 111% |
| **3B** | **2.94B** | **36,383** | **254,681** | **6** | **47.3 GB** | **55.0B** | **58.9B** | **93%** ✅ |

> **결론:** tok/s가 per-GPU임을 감안하면, 60시간 내에 1B~2.5B는 Chinchilla를 크게 초과(과잉학습)합니다. **3B가 Chinchilla ~93%로 compute budget에 가장 효율적으로 맞는 최적 규모**입니다.

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

*EVAFRILL-Mo — 밑바닥부터, selective scan 하나하나 직접 쌓아올린 3B 하이브리드 모델.*

</div>
