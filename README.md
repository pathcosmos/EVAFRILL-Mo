> 한국어 | **[English](README.en.md)**

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
![Model](https://img.shields.io/badge/Model-2.94B%20params-blue)
![Training](https://img.shields.io/badge/Training-319K%20steps-brightgreen)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-pathcosmos%2FEVAFRILL--Mo--3B-yellow)](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

NVIDIA [Nemotron-H](https://arxiv.org/abs/2504.03624) 아키텍처에서 영감을 받아 밑바닥부터 직접 구현한 **30억 파라미터 하이브리드 Mamba-2 + Transformer** 언어 모델입니다. 7× NVIDIA B200 GPU에서 60시간 Chinchilla-optimal 사전학습을 목표로 설계되었습니다.

**모델 다운로드**: [🤗 HuggingFace Hub](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

</div>

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [아키텍처](#아키텍처)
- [Nemotron-Nano 단편화 도입](#nemotron-nano-아키텍처-단편화-도입)
- [하드웨어 환경](#하드웨어-환경)
- [프로젝트 구조](#프로젝트-구조)
- [빠른 시작](#빠른-시작)
- [적용 기술 상세](#적용-기술-상세)
- [1B → 3B 전환 경위](#1b--3b-전환-경위)
- [3B 하드웨어 제약 최적화](#3b-하드웨어-제약-최적화)
- [학습 데이터](#학습-데이터)
- [개발 히스토리](#개발-히스토리)
- [SFT (Supervised Fine-Tuning)](#sft-supervised-fine-tuning)
- [모델 정렬 및 평가 (Model Alignment & Evaluation)](#모델-정렬-및-평가-model-alignment--evaluation)
  - [SFT 모델 평가 결과](#sft-모델-평가-결과)
  - [Preference 데이터 준비](#preference-데이터-준비)
  - [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
  - [종합 평가 결과](#종합-평가-결과)
  - [ORPO 비교 실험 (2026-03-25)](#orpo-비교-실험-2026-03-25)
  - [배포 및 추론](#배포-및-추론)
  - [향후 개선 방향](#향후-개선-방향)
- [부록: 실행 가이드](#부록-실행-가이드)
- [벤치마크 결과](#벤치마크-결과)
- [관련 프로젝트](#관련-프로젝트)
- [참조 논문](#참조-논문)
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

### 3B 모델 구성 (학습 완료)

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
레이어 13-23: Mamba-2 SSM ×11  ──┘
레이어 24:    Attention (GQA)        후반부
레이어 25:    Mamba-2 SSM ×1
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
| GQA (Grouped Query Attention) | `n_kv_heads=8` (비율 3:1) | ✅ 도입 |
| FP8 네이티브 학습 | TransformerEngine MXFP8BlockScaling | ✅ 도입 |
| 대규모 d_state (128) | `mamba_d_state=128` | ✅ 도입 |
| 청크 기반 selective scan | `mamba_chunk_size=256` | ✅ 도입 |
| MoE (Mixture of Experts) | — | ❌ 포기 (소규모에서 효과 미미) |
| Knowledge Distillation | — | ❌ 포기 (teacher 모델 부재) |
| RLHF/DPO 파이프라인 | Native DPO + LoRA (TRL 미사용) | ✅ 도입 (Post-SFT) |
| 4B/8B 규모 | 2.94B로 축소 | 🔄 스케일 조정 |
| 수조 토큰 학습 | 55B 토큰 (~1.34 에포크, Chinchilla 93%) | 🔄 스케일 조정 |

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
│   ├── layers.py              # RMSNorm, SwiGLU, 임베딩
│   └── lora.py                # LoRA 어댑터 (Attention + Mamba 레이어)
│
├── train/                     # 학습
│   ├── pretrain.py            # 사전학습 엔트리포인트
│   ├── trainer.py             # 학습 루프 (DDP, FP8, 체크포인팅)
│   ├── sft.py                 # 지도 미세조정 (SFT)
│   ├── dpo.py                 # DPO 선호도 학습 (Native, LoRA)
│   ├── orpo.py                # ORPO 선호도 최적화 (TRL 기반)
│   ├── orpo_native.py         # ORPO 네이티브 구현 (TRL 미사용, 실제 학습에 사용)
│   └── utils.py               # Cosine 스케줄러, DDP 설정, 체크포인트 유틸
│
├── data/                      # 데이터 파이프라인
│   ├── dataset.py             # PackedDataset (memmap + MADV_WILLNEED 힌트)
│   ├── prepare.py             # 토큰화 파이프라인
│   ├── prepare_sft_data.py    # SFT 데이터 준비
│   ├── filter_sft_v2.py       # SFT 데이터 품질 필터링
│   ├── sft_dataset.py         # SFT 대화형 데이터셋
│   ├── dpo_dataset.py         # DPO 선호도 쌍 데이터셋
│   ├── prepare_preference_combined.py  # 7개 preference 소스 → 통합 JSONL
│   ├── generate_repetition_preference.py  # 반복 억제 preference 데이터 생성
│   └── *.bin                  # 바이너리 토큰 파일 (저장소에 미포함)
│
├── eval/                      # 평가
│   ├── evafrill_eval.py       # 종합 4-phase 평가 (PPL, 생성, 보정, lm-eval)
│   ├── full_eval_pipeline.py  # 전체 평가 파이프라인 오케스트레이션
│   ├── perplexity.py          # 퍼플렉시티 평가
│   ├── generate.py            # 텍스트 생성 / 샘플링
│   ├── comprehensive_eval.py  # 종합 평가 도구
│   └── report_generator.py    # 마크다운 평가 리포트 생성
│
├── scripts/                   # 실행, 모니터링, 배포 스크립트
│   ├── merge_checkpoints.py   # SLERP/LERP 체크포인트 보간 (alignment tax 완화)
│   ├── export_to_hf.py        # HuggingFace Hub 모델 내보내기 + push
│   ├── convert_to_hf.py       # 네이티브 → HuggingFace 포맷 변환
│   └── migrate_qkv_checkpoint.py  # QKV 체크포인트 레이아웃 마이그레이션
│
├── configs/                   # YAML 학습 설정 파일
├── benchmarks/                # 처리량 & 프로파일링 도구
├── tokenizer/                 # SentencePiece 토크나이저 학습
├── reports/                   # 평가 및 분석 리포트
├── docs/                      # 하드웨어 & 환경 문서
├── train_3b_sft_1gpu.sh       # H100 MIG SFT 런치 스크립트
├── train_3b_dpo_1gpu.sh       # H100 MIG DPO 런치 스크립트
├── train_3b_orpo_1gpu.sh      # H100 MIG ORPO 런치 스크립트
├── requirements.txt           # Python 의존성 목록
├── README.en.md               # 영문 README
└── demo/app.py                # Gradio 데모 서버
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

### 추론 예제 (Python)

```python
import torch
from model.transformer import LLM
from tokenizers import Tokenizer

# 모델 로드 (SLERP 권장)
model = LLM.from_pretrained("checkpoints/3b_dpo/checkpoint-slerp")
model = model.to(device="cuda:0", dtype=torch.bfloat16)
model.eval()

tok = Tokenizer.from_file("tokenizer/korean_sp/tokenizer.json")

# Chat template 적용
prompt = "<|user|>\n인공지능이란 무엇인가요?\n<|assistant|>\n"
ids = torch.tensor([tok.encode(prompt).ids], device="cuda:0")

# 생성 (권장: temp=0.7, rep_penalty=1.2)
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

> 💡 **Gradio 데모**: `python3 demo/app.py` 실행 후 http://localhost:7860 접속
>
> 📦 **HuggingFace**: [pathcosmos/EVAFRILL-Mo-3B](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)에서 모델 다운로드

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
| **GQA (Grouped Query Attention)** | 24개 쿼리 헤드, 8개 KV 헤드 (3:1 비율). KV 캐시 메모리 67% 절감 | `model/attention.py:77` |
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

### 1단계 — [FRANKENSTALLM](https://github.com/pathcosmos/FRANKENSTALLM) (순수 Transformer)

순수 Transformer decoder-only LLM으로 시작했습니다 (Frankenstein + LLM). 한국어 + 영어 + 코드 + 수학 데이터로 커스텀 SentencePiece 토크나이저를 학습시켰으며 (어휘 64,000), 기본 학습 파이프라인(DDP, 체크포인트, cosine 스케줄러)을 구축했습니다. 해당 프로젝트의 전체 코드와 문서는 [FRANKENSTALLM GitHub 저장소](https://github.com/pathcosmos/FRANKENSTALLM)에서 확인할 수 있습니다.

### 2단계 — 11단계 구현 계획 (전체 완료)

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

### 3단계 — Nemotron-Nano 단편화 도입 & 최적 규모 탐색 (EVAFRILL-Mo)

핵심 질문: **65시간 × 7 B200에서 Chinchilla-optimal 학습이 가능한 최대 모델 크기는?**

- Nemotron-Nano의 핵심 설계 원칙을 추출하여 5개 규모(1B~3B)에 적용 (상세: [단편화 도입 섹션](#nemotron-nano-아키텍처-단편화-도입))
- 5개 모델 체계적 벤치마크 (각 20 steps, 7 GPU)
- **Mamba Memory Cliff 현상 발견**: 배치 크기 임계점에서 ~7.5배 메모리 점프
- **1B 모델 최종 선정**: 유일한 Chinchilla-optimal 후보 (107% 달성)

### 4단계 — VectorDB / Memory DB 조사

LLM 사전학습에 vectorDB나 memoryDB가 도움이 되는지 조사했습니다:

| 접근법 | 조사 결과 | 판정 |
|--------|----------|------|
| RETRO 스타일 검색 증강 학습 | Mamba에 적용 불가 — CCA 레이어가 Transformer 전용 아키텍처 | ❌ 불가 |
| LMDB/RocksDB 데이터 로딩 | 2.2TB RAM에 82GB 데이터 전부 캐싱됨 → 개선 없음 | ❌ 불필요 |
| Curriculum Learning (DB 기반) | DB 없이도 가능, 1-3% 개선 수준 | ❌ DB 불필요 |
| FAISS/Milvus/LanceDB | 미설치 상태, 도입 오버헤드 과대 | ❌ 비용 초과 |

**결론:** 65시간 마감 하에서 구현 오버헤드가 학습 시간을 잠식하므로 도입 비추천. 순수 사전학습에 집중하는 것이 최선.

### 5단계 — 1B 학습 시작 & 과잉학습 감지

- **모델**: 994M 파라미터, 18층 (Mamba-2 16개 + Attention 2개)
- **학습 시작**: 45,776 스텝, batch=16, ~90,000 tok/s (per-GPU)
- **감지**: step 3,700 시점에서 전체 소요 시간이 ~9.3시간으로 예측됨
- **원인 분석**: tok/s가 per-GPU 값임을 확인 → 실제 aggregate는 630,000 tok/s
- **판단**: 1B에 65시간 투자 시 Chinchilla의 7.5배 과잉학습 → compute 낭비
- **결정**: step 4,230에서 1B 학습 중단, 3B 규모로 전환

### 6단계 — 3B 사전학습 완료

- **모델**: 2,944M 파라미터, 26층 (Mamba-2 24개 + Attention 2개)
- **벤치마크**: batch=6~12까지 순차 테스트 → batch=6이 Memory Cliff 직전 최대값
- **처리량**: 36,383 tok/s (per-GPU), 254,681 tok/s (aggregate)
- **학습**: 319,772 스텝, ~55B 토큰, ~60시간
- **Chinchilla 달성률**: ~93% (1.34 에포크)
- **체크포인트**: 1,000 step마다 자동 저장 (model + optimizer + scheduler + train_state)
- **복구 래퍼**: `train_3b_resilient.sh` — 크래시 시 최신 체크포인트에서 자동 재시작 (최대 10회, 포트 자동 변경)
- **완료**: 2026-03-09, 319,772 step 전부 완료. 최종 체크포인트 `checkpoints/3b_final/checkpoint-0319772`

#### 사전학습 Loss 추이 (25k 구간 평균)

| 구간 | 평균 Loss | 변화 |
|------|--------:|------|
| 0~25k | 2.96 | 초기 수렴 |
| 25~50k | 4.77 | epoch 전환 spike |
| 50~100k | 2.39 | 급격한 감소 |
| 100~150k | 2.00 | 안정적 감소 |
| 150~200k | 1.87 | 점진적 감소 |
| 200~250k | 1.77 | 점진적 감소 |
| 250~319k | 1.69 | 수렴 완료 |

### 7단계 — 3B SFT v2 (Early Stop으로 완료)

사전학습 완료된 3B 모델 위에 한국어 SFT(Supervised Fine-Tuning)를 수행했습니다.

#### 환경 전환: B200 8GPU → H100 MIG 1GPU

B200 클러스터 반납 후 H100 MIG 3g.40gb 단일 파티션 환경으로 전환했습니다.

| 항목 | B200 8GPU (사전학습) | H100 MIG (SFT) |
|------|---------------------|-----------------|
| GPU | 8× B200 (183GB each) | 1× H100 MIG 3g.40gb (~42GB) |
| 정밀도 | FP8 (MXFP8) | BF16 + Gradient Checkpointing |
| 배치 | bs=6 × 7GPU = 42 | bs=4, grad_accum=7, eff=28 |
| 속도 | 0.67s/step | 6.8s/step |

#### SFT 학습 설정

| 파라미터 | 값 |
|---------|-----|
| 베이스 체크포인트 | `checkpoints/3b_final/checkpoint-0319772` |
| SFT 데이터 | `data/sft_combined/train_filtered.jsonl` |
| Validation 데이터 | `data/sft_combined/val_filtered.jsonl` |
| 설정 파일 | `configs/h100_mig/korean_3b_sft_1gpu.yaml` |
| 런치 스크립트 | `train_3b_sft_1gpu.sh` (resilient wrapper) |
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
| 정밀도 | BF16 + Gradient Checkpointing |
| VRAM 사용 | 24.0GB / 40.3GB (60%) |
| 토크나이제이션 | 초기화 시 전체 pre-tokenize + 캐시 |

#### SFT Validation Loss 추이 — 수렴 및 Early Stop 근거

| Step | val_loss | Δval_loss | 구간 |
|-----:|--------:|---------:|------|
| 5,000 | 1.8774 | — | 급감기 |
| 10,000 | 1.8424 | -0.0350 | |
| 15,000 | 1.8239 | -0.0185 | |
| 20,000 | 1.8124 | -0.0115 | 감속기 |
| 25,000 | 1.8050 | -0.0074 | |
| 30,000 | 1.8001 | -0.0049 | |
| 35,000 | 1.7968 | -0.0033 | |
| 40,000 | 1.7949 | -0.0019 | Plateau 진입 |
| 45,000 | 1.7940 | -0.0009 | |
| 50,000 | 1.7933 | -0.0007 | |
| 55,000 | 1.7928 | -0.0005 | |
| 60,000 | 1.7928 | -0.0000 | 정체 |
| **65,000** | **1.7924** | **-0.0004** | **Early Stop 결정** |

13회 연속 best 갱신이나, 50K 이후 개선이 측정 노이즈 수준으로 감소.

#### Early Stop 결정 (Step 65,000 / 135,000, 48.15%)

**결정일**: 2026-03-22
**최종 best val_loss**: 1.7924 (step 65,000)
**최종 체크포인트**: `checkpoints/3b_sft_v2/checkpoint-best`, `checkpoint-0065059` (emergency)

**중단 근거 — 수학적 분석:**

1. **Asymptote 도달**: 지수 감쇠 피팅(`L = a·exp(-b·t) + c`) 결과, 이론적 최저 val_loss(c) = ~1.7922. 현재 1.7924로 이미 asymptote에 거의 도달 (R² = 0.9994)
2. **개선량 소멸**: 50K→65K (15,000 steps, ~28시간) 총 개선 0.0009. 남은 70K steps(~5.5일) 예상 개선 0.001~0.003
3. **PPL 차이 무의미**: val_loss 0.001 차이 = PPL 6.006 → 6.000 (ΔPPL = 0.006). 실제 출력 품질에 체감 불가
4. **SNR 부족**: 5K-step 단위 측정 노이즈(σ=0.0003) 대비 예상 개선량(0.0002)의 SNR = 0.57σ — 통계적으로 유의하지 않음

**중단 근거 — 실용적 분석:**

1. **기회비용**: 동일 GPU 시간으로 정량 평가(KoBEST/KLUE), 데이터 재구성 후 새 SFT, 또는 DPO/RLHF 수행이 기대 수익 훨씬 높음
2. **Overfitting 없음**: 전 구간에서 val-train gap이 0.01~0.03 범위로 안정. 단조 증가 없음
3. **Cosine LR 후반부 효과 소진**: lr이 이미 peak의 53%로 감소, 후반부 급격한 개선 가능성 없음

#### SFT 학습 안정성 지표

| 지표 | 값 | 판정 |
|------|-----|------|
| 최대 gnorm | 4.219 (warmup step 140) | 정상 |
| gnorm > 5 | 0건 | 안전 |
| nan/inf/OOM | 0건 | 안전 |
| 메모리 | 24.0GB 전 구간 고정 | 안정 |
| tok/s 추세 | 평균 5,343, 시간 경과에 따른 감소 없음 | 안정 |
| SIGTERM 복구 | step 421에서 1회, 정상 재개 | 정상 |
| epoch | 0 (단일 epoch, 데이터 반복 없음) | 정상 |

---

## SFT (Supervised Fine-Tuning)

### 개요

사전학습 완료된 3B 모델(`checkpoints/3b_final/checkpoint-0319772`)을 한국어 instruction-following 데이터로 SFT를 수행했습니다. H100 MIG 3g.40gb 단일 GPU 환경에서 진행했으며, 수렴 분석을 통해 step 65,000에서 early stop했습니다.

### SFT 데이터

| 항목 | 값 |
|------|-----|
| 학습 데이터 | `data/sft_combined/train_filtered.jsonl` |
| 검증 데이터 | `data/sft_combined/val_filtered.jsonl` |
| 형식 | 대화형 (conversation) JSONL |
| 토크나이제이션 | 초기화 시 전체 pre-tokenize + `.sft_cache_*.pt` 캐시 |

### 핵심 기법

| 기법 | 설명 |
|------|------|
| **NEFTune** (alpha=5.0) | Embedding에 uniform noise 주입으로 일반화 성능 향상 (Jain et al., 2023) |
| **Dynamic Padding** | 배치 내 최대 시퀀스 길이에 맞춰 패딩, 64 정렬. 고정 길이 대비 연산 낭비 감소 |
| **Gradient Checkpointing** | Activation 재계산으로 VRAM 절약. MIG 42GB 제약 하에서 3B 모델 학습 가능 |
| **Cosine LR Decay** | Peak 7.0e-06에서 cosine 감쇠. 사전학습 lr(3e-4)의 1/43 수준으로 보수적 설정 |
| **Resilient Wrapper** | `train_3b_sft_1gpu.sh` — SIGTERM/크래시 시 자동 체크포인트 저장 및 재시작 |

### 결과 요약

```
학습 기간:    2026-03-17 ~ 2026-03-22 (5일)
진행 steps:   65,000 / 135,000 (48.15%)
최종 val_loss: 1.7924 (13회 연속 best 갱신)
중단 사유:    Plateau — asymptote 도달, 추가 학습의 기대 수익 < 측정 노이즈
체크포인트:   checkpoints/3b_sft_v2/checkpoint-best (step 65,000)
```

### 수렴 분석 시각화

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

- **급감기** (5K~20K): val_loss 1.877 → 1.812, Δ = -0.065
- **감속기** (20K~35K): val_loss 1.812 → 1.797, Δ = -0.015
- **Plateau** (35K~65K): val_loss 1.797 → 1.792, Δ = -0.005 (개선폭이 noise 수준)

---

## 모델 정렬 및 평가 (Model Alignment & Evaluation)

SFT v2 완료(step 65,000) 후의 내러티브: **SFT 모델 품질 평가 → DPO 선호도 학습 → SLERP 병합 → 종합 평가 → ORPO 비교 실험**.

### SFT 모델 평가 결과

`eval/evafrill_eval.py`를 사용한 4-phase 평가 체계 중 Phase 2(생성 품질)를 완료했습니다.

**평가 환경**: H100 MIG 3g.40gb, batch_size=2

| Phase | 설명 | 상태 | 비고 |
|-------|------|------|------|
| Phase 1 (PPL) | Perplexity on 3b_val.bin | ⏭ 스킵 | stride=2048으로도 ~4.4시간 소요, 우선순위 낮음 |
| Phase 2 (생성) | 15 프롬프트 × 4 디코딩 설정 | ✅ 완료 | ~2.5시간 |
| Phase 3 (보정) | Calibration curve | ⏭ 스킵 | |
| Phase 4 (lm-eval) | 6개 벤치마크 (kmmlu 등) | ❌ 중단 | kmmlu가 ~167,000 문제 포함, H100 MIG에서 12-18시간 소요로 DPO에 GPU 할당 |

**체크포인트**: `checkpoints/3b_sft_v2/checkpoint-best` (step 65,059)

| 프롬프트 | Greedy 3-gram 반복률 | 평가 |
|----------|--------------------:|------|
| 대한민국의 수도는 | 96.85% | 동일 문구 반복 루프 |
| 양자 컴퓨터란 | 96.85% | 심각한 반복 |
| 건강한 식습관을 위해서는 | 59.45% | 상대적으로 양호 |
| 인공지능이란 | 50.00% | 구조화된 목록 형식이나 반복 존재 |
| 한국어는 세계에서 | 35.83% | 낮은 반복이나 한영 혼합 깨짐 |
| **평균** | **~76%** | **DPO로 반복 문제 해결 필요** |

**핵심 발견**: SFT 모델은 한국어 텍스트를 생성하나, greedy 디코딩에서 심각한 반복 루프 발생. DPO를 통한 선호도 학습이 반복 억제에 필수적.

---

### Preference 데이터 준비

`data/prepare_preference_combined.py`로 7개 한국어 preference 데이터셋을 통합 JSONL로 변환했습니다.

| 데이터셋 | 레코드 수 | 형식 |
|----------|----------:|------|
| heegyu/orca-math-korean-preference-cleaned | 192,422 | chosen/rejected |
| nayohan/preference-collection-ko-full | 199,577 | orig_response_A/B + orig_preference |
| kuotient/orca-math-word-problems-193k-korean | 192,375 | chosen/rejected |
| FreedomIntelligence/alpaca-gpt4-korean | 49,969 | chosen/rejected |
| heegyu/orca_ko | 42,989 | chosen/rejected |
| HAERAE-HUB/KOFFQA-GuardInstruct-v1 | 7,210 | chosen/rejected |
| jojo0217/korean_rlhf_dataset | 0 | SFT 전용 (preference pair 없음) |
| **합계** | **684,542 → 504,103** | 토크나이징 후 유효 샘플 수 |

---

### DPO (Direct Preference Optimization)

#### DPO vs ORPO: 기법 비교 및 선택 근거

| | DPO | ORPO |
|---|---|---|
| **Reference 모델** | 필요 (SFT 모델의 logprob) | **불필요** |
| **VRAM** | 높음 (ref model forward 추가) | 낮음 |
| **손실 함수** | `log σ(β · (Δchosen - Δrejected))` | SFT loss + λ · odds ratio penalty |
| **학습 단계** | SFT → DPO (2단계) | **SFT와 동시** (1단계) |
| **성숙도** | 표준, 검증 많음 | 비교적 신생 (2024) |

**이 프로젝트에서 DPO를 선택한 이유:**

1. **SFT가 이미 완료됨** — SFT v2가 step 65,000에서 수렴 완료. ORPO를 쓰려면 SFT를 처음부터 재실행해야 하므로 5일간의 학습이 낭비됨
2. **LoRA B-zeroing으로 VRAM 단점 해소** — lora_B를 임시 0으로 만들어 ref logprob 계산, 모델 복제 없이 실측 6.3GB 동작
3. **Nemotron-H 논문이 DPO 사용** — 2-round DPO + SLERP merge 파이프라인을 표준으로 채택

> **참고:** `train/orpo.py`가 프로젝트에 존재하며, 향후 처음부터 재설계 시 ORPO로 SFT+정렬을 한번에 수행하는 것이 더 효율적일 수 있음.

#### 학습 설정

**설계 결정:**

| 결정 | 선택 | 근거 |
|------|------|------|
| **프레임워크** | Native DPO (TRL 미사용) | TRL은 HF AutoModel 필요 — Hybrid Mamba 미지원 |
| **파라미터 효율화** | LoRA (rank=32, alpha=64) | ~22GB VRAM → H100 MIG 42GB에 여유 있게 적합 |
| **Reference 모델** | LoRA B-zeroing | lora_B를 임시 0으로 만들어 ref logprob 계산, 모델 복제 불필요 |
| **체크포인트 병합** | SLERP interpolation | `slerp(W_sft, W_dpo, α=0.5)`로 alignment tax 완화 |

**LoRA 어댑터:**

```
적용 레이어:    Attention (qkv_proj, out_proj) + Mamba-2 (in_proj, out_proj)
어댑터 수:      52개
학습 파라미터:  21,438,464 (전체 2.97B의 0.72%)
VRAM 사용:     ~6.3GB (MIG 42GB의 15%)
```

**2-Round 전략 (Nemotron-H 스타일):**

| | Round 1 (Exploration) | Round 2 (Exploitation) |
|---|---------|---------|
| **목적** | 광범위 선호도 학습 | 정밀 조정, over-alignment 방지 |
| **데이터** | 전체 preference (504K 샘플) | 동일 |
| **Steps** | 3,000 | 2,000 |
| **Beta** | 0.1 | 0.05 |
| **LR** | 5e-7 | 1e-7 |
| **Warmup** | 100 steps | 50 steps |
| **Batch** | bs=1 × grad_accum=16 = eff 16 | 동일 |

#### 학습 결과

**Round 1** (2026-03-23, 4시간 33분):

```
step   10 | loss 0.6941 | margin -0.006 | lr 5.0e-08  (warmup)
step  500 | loss 0.6543 | margin  0.120 | lr 4.93e-07
step 1500 | loss 0.6012 | margin  0.210 | lr 2.50e-07
step 3000 | loss 0.5652 | margin  0.245 | lr 5.0e-08   (최종)
```

Loss 0.693 → 0.565 (18.5% 하락), Margin +0.245: chosen/rejected 구분 학습 완료. 체크포인트: `checkpoints/3b_dpo_r1/checkpoint-merged`

**Round 2** (2026-03-23, 3시간 2분):

```
step   50 | loss 0.6953 | margin  0.003 | lr 1.0e-07  (warmup 완료)
step 1000 | loss 0.6906 | margin  0.008 | lr 5.7e-08
step 2000 | loss 0.6886 | margin -0.005 | lr 1.0e-08  (최종)
```

Loss 0.692 → 0.689 (0.5% 변화): 의도적으로 완만한 하강 — 보수적 미세 조정이 설계대로 작동. gnorm 1.6~2.2로 Round 1(3~4)보다 안정적. 체크포인트: `checkpoints/3b_dpo_r2/checkpoint-merged`

#### SLERP 병합 및 최종 모델 선택

**SLERP(Spherical Linear Interpolation)란?** DPO가 선호도를 학습하는 과정에서 SFT 지식을 일부 잃는 alignment tax를 완화하기 위해, 두 체크포인트를 가중치 공간에서 구면 보간합니다. 일반 LERP와 달리 가중치 벡터의 방향을 보존하므로 각 체크포인트의 특성을 더 잘 유지합니다.

```
SLERP(W_sft, W_dpo, α=0.5):
  α=0: 순수 SFT  |  α=0.5: SFT 50% + DPO 50% (Nemotron-H 기본값)  |  α=1: 순수 DPO
```

**3-체크포인트 비교 (2026-03-24, 15개 프롬프트 greedy 반복률):**

| 모델 | 평균 반복률 | 최저 반복 프롬프트 수 |
|------|:---------:|:------------------:|
| SFT v2 | 79.8% | 1/15 |
| DPO Round 2 | 80.7% | 1/15 |
| **SLERP (α=0.5)** | **74.5%** | **7/15** |

**최종 선택: SLERP (α=0.5)** — `checkpoints/3b_dpo/checkpoint-slerp`

SLERP가 15개 프롬프트 중 7개에서 최저 반복률 달성. "한국의 전통 음식" 90.9% → 39.4% (-51.5pp) 등 큰 개선 사례 존재.

**lm-eval 3-way 비교 (limit=100, 0-shot):**

| Benchmark | SFT | DPO R2 | SLERP |
|-----------|----:|-------:|------:|
| hellaswag | 39.0% | 39.0% | 39.0% |
| belebele_kor_Hang | 30.0% | 29.0% | 30.0% |
| arc_easy | 28.0% | 28.0% | 27.0% |
| arc_challenge | 21.0% | 22.0% | 22.0% |
| global_mmlu_full_ko | 23.4% | 23.4% | 23.3% |

3개 체크포인트 간 accuracy 차이가 1% 이내 — LoRA 기반 DPO + SLERP가 지식을 거의 완벽하게 보존함 (alignment tax 미미).

---

### 종합 평가 결과

#### 생성 품질 비교 (Greedy 반복률)

15개 프롬프트별 Greedy 3-gram 반복률 전체 비교:

| 프롬프트 | SFT | DPO R2 | SLERP | 최저 |
|----------|----:|-------:|------:|------|
| 대한민국의 수도는 | 85.0 | 89.4 | 96.9 | SFT |
| 인공지능이란 | 61.8 | 61.8 | **50.0** | SLERP |
| 한국의 전통 음식 중에서 | 90.9 | 74.8 | **39.4** | SLERP |
| 지구 온난화의 주요 원인은 | 82.3 | 87.4 | **72.4** | SLERP |
| 프로그래밍을 배우려면 | 89.0 | 89.0 | 90.6 | SFT/DPO |
| 조선시대에는 | 65.0 | 84.3 | **65.0** | SFT=SLERP |
| 물리학에서 에너지란 | 88.6 | 93.7 | **86.6** | SLERP |
| 한국어는 세계에서 | 65.8 | 65.8 | **52.0** | SLERP |
| 경제 성장을 위해서는 | 77.2 | 77.2 | **70.5** | SLERP |
| 우주 탐사의 역사를 보면 | 95.3 | 95.3 | 95.3 | 동률 |
| 머신러닝과 딥러닝의 차이는 | 89.4 | 89.4 | **83.1** | SLERP |
| 한국 문학의 대표적인 작품으로는 | 74.0 | **72.8** | 85.4 | DPO |
| 양자 컴퓨터란 | 96.9 | 96.9 | 96.9 | 동률 |
| 건강한 식습관을 위해서는 | 56.3 | **55.9** | 55.9 | DPO=SLERP |
| 세계 2차 대전 이후 | 79.5 | 77.6 | 77.6 | DPO=SLERP |
| **평균** | **79.8%** | **80.7%** | **74.5%** | **SLERP** |

**한계 — 솔직한 평가:**
- 목표였던 30% 이하에는 크게 못 미침 (평균 74.5%)
- DPO 단독은 SFT보다 오히려 미세하게 악화 (80.7% vs 79.8%) — DPO가 반복 문제를 직접 해결하지 못함
- 근본 원인: 3B 규모 하이브리드 Mamba 모델의 greedy 반복은 모델 아키텍처 수준의 문제일 가능성

#### Repetition Penalty 디코딩 테스트

SLERP 모델에 `repetition_penalty=1.2`를 적용하면 반복률이 극적으로 감소합니다:

| 프롬프트 | greedy (r=1.0) | greedy (r=1.2) | t0.7 + r1.2 |
|----------|:--------------:|:--------------:|:-----------:|
| 대한민국의 수도는 | 81.5% | **13.4%** | **0.4%** |
| 인공지능이란 | 61.8% | **13.4%** | **1.6%** |
| 한국의 전통 음식 중에서 | 74.8% | **0.0%** | **0.0%** |
| 건강한 식습관을 위해서는 | 66.1% | **0.8%** | **1.2%** |
| 한국어는 세계에서 | 48.0% | 0.0% | 0.0% |
| **평균** | **66.4%** | **~5.5%** | **~0.6%** |

**생성 품질 예시 (greedy + r=1.2):**

```
[대한민국의 수도는] → 서울특별시이고, 그 외 지역은 광역시로 분류한다.
  대한민국의 행정구역 변천사 1945년 8월 15일 - 경기도 인천부(仁川府)...

[한국의 전통 음식 중에서] → 가장 유명한 것이 바로 김치이다. 김치는
  한국인의 주식이자, 세계인에게 사랑받는 국민음식으로 자리 잡았다...

[건강한 식습관을 위해서는] → 균형 잡힌 식단이 중요하다. 특히, 단백질은
  필수 아미노산으로 구성돼 있어 체내 흡수율이 높아 건강에 좋다...
```

**권장 추론 설정**: `temperature=0.7, repetition_penalty=1.2`

#### lm-eval 벤치마크

**limit=100 3-way 비교** (0-shot, kmmlu 제외):

| Benchmark | SFT | DPO R2 | SLERP |
|-----------|----:|-------:|------:|
| hellaswag | 39.0% | 39.0% | 39.0% |
| belebele_kor_Hang | 30.0% | 29.0% | 30.0% |
| arc_easy | 28.0% | 28.0% | 27.0% |
| arc_challenge | 21.0% | 22.0% | 22.0% |
| global_mmlu_full_ko | 23.4% | 23.4% | 23.3% |

**limit=500 최종 SLERP 모델** (0-shot, kmmlu 제외):

| Benchmark | Accuracy | Random | 비고 |
|-----------|:--------:|:------:|------|
| hellaswag | **34.6%** | 25.0% | 영어 상식 추론 +9.6pp |
| arc_easy | **32.0%** | 25.0% | 기초 과학 +7.0pp |
| global_mmlu_full_ko | 23.7% | 25.0% | 한국어 전문지식 미흡 |
| belebele_kor_Hang | 23.6% | 25.0% | 한국어 독해 제한적 |
| arc_challenge | 18.2% | 25.0% | 고난이도 추론 부족 |

hellaswag/arc_easy에서만 random 대비 유의미한 개선. 한국어 벤치마크는 random 수준 — 3B 모델 규모와 학습 데이터 양의 근본적 제약.

#### 다중 α SLERP 실험

α=0.3, 0.5, 0.7로 SLERP 병합 후, 5개 프롬프트 + greedy + rep_penalty=1.2로 비교:

| α | SFT 비중 | DPO 비중 | 평균 반복률 |
|---|---------|---------|-----------|
| 0.3 | 70% | 30% | 9.1% |
| **0.5** | **50%** | **50%** | **5.8%** (최저) |
| 0.7 | 30% | 70% | 7.3% |

α=0.5(Nemotron-H 기본값)가 최적.

#### 정성적 Chat Template 평가

SLERP(α=0.5) 모델, chat template(`<|user|>\n...\n<|assistant|>\n`), `temperature=0.7, repetition_penalty=1.2`:

```
USER: 김치를 만드는 방법을 간단히 알려주세요.
ASSISTANT: 김치는 한국 요리에서 중요한 역할을 하는 음식으로, 녹색 채소가
  주재료인 발효 식품이랍니다!...

USER: 인공지능이 무엇인지 쉽게 설명해주세요.
ASSISTANT: 인공지능(artificial intelligence, AI)은 기계가 스스로 학습하고,
  의사 결정 능력을 가지고 있는 컴퓨터의 한 종류로...
```

| 항목 | 평가 | 비고 |
|------|------|------|
| **반복** | ✅ 양호 | rep_penalty=1.2로 반복 거의 없음 |
| **유창성** | ✅ 양호 | 자연스러운 한국어 문장 생성 |
| **Instruction following** | ⚠️ 보통 | 질문에 답하려는 시도는 하나, 정확도 낮음 |
| **사실 정확도** | ❌ 낮음 | "세종 재위 100년 초과" 등 hallucination 발생 |
| **코드 생성** | ❌ 실패 | 피보나치 코드 요청에 의미 없는 응답 |

유창한 한국어 생성은 가능하나 사실 정확도와 복잡한 추론은 3B 규모의 근본적 제약.

---

### ORPO 비교 실험 (2026-03-25)

#### 동기 및 설정

DPO가 반복 문제를 직접 해결하지 못했음 (SFT 79.8% → DPO 80.7%, 오히려 악화). ORPO는 SFT+정렬을 동시에 학습하므로, 분리 파이프라인의 구조적 한계를 극복할 수 있는지 검증.

ORPO(Hong et al., 2024): `L_ORPO = L_SFT + λ * L_OR` — reference model 없이 SFT loss와 odds ratio penalty를 단일 목적함수로 결합.

기존 `train/orpo.py`는 TRL 기반(HF AutoModel 필요)이므로, DPO와 동일한 이유로 `train/orpo_native.py`를 네이티브 구현.

| 항목 | 값 |
|------|-----|
| 시작점 | `checkpoints/3b_final/checkpoint-0319772` (Pretrained) |
| 데이터 | 504,103 preference pairs |
| Steps | 10,000 |
| LR | 5e-6 |
| λ | 1.0 |
| LoRA | rank=32, alpha=64 |
| VRAM | 6.2GB |
| 소요 시간 | 12시간 48분 |

#### 학습 결과

```
step     10 | sft 10.16 | or 0.909 | total 11.07
step  1,000 | sft  6.25 | or 0.751 | total  7.00
step  5,000 | sft  6.03 | or 0.565 | total  6.60
step 10,000 | sft  5.85 | or 0.558 | total  6.41
```

SFT loss -42.4%, OR loss -38.6% 하강.

#### 종합 비교 및 결론

| 지표 | SLERP (α=0.5) | ORPO (10K) | 승자 |
|------|:-------------:|:----------:|:----:|
| **Greedy 반복률** | **74.5%** | 87.1% | SLERP |
| **greedy+r1.2 반복률** | 5.5% | **3.7%** | ORPO |
| **t0.7+r1.2 반복률** | **0.6%** | 1.8% | SLERP |
| **hellaswag** | **39.0%** | 35.0% | SLERP |
| **arc_easy** | 27.0% | **30.0%** | ORPO |
| **belebele_kor** | **30.0%** | 23.0% | SLERP |
| **arc_challenge** | 22.0% | **19.0%** | SLERP |
| **global_mmlu_ko** | 23.3% | 23.3% | 동률 |
| **Chat 대화 품질** | ✅ 유창 | ❌ 깨짐 | **SLERP** |
| **학습 시간** | 5일+8시간 | **12.8시간** | ORPO |

**결론: SLERP 승리 (현재 설정에서).**

ORPO 열세의 핵심 이유: SFT 학습 부족. ORPO의 SFT loss가 5.85에서 멈춘 반면, SFT v2 val_loss는 1.79 — ORPO 10,000 steps는 SFT 65,000 steps에 비해 절대적으로 부족. 공정한 비교를 위해서는 ORPO를 65,000 steps 이상 학습해야 하며(약 5일), 현재 실험은 탐색적 성격.

- ORPO의 **시간 효율성**은 매력적이나, 동등한 SFT 품질을 달성하려면 결국 비슷한 steps가 필요
- **SFT loss가 충분히 수렴한 후에야** OR loss의 정렬 효과가 발휘될 것으로 예상
- 현재 SLERP 파이프라인이 이 모델/데이터 조합에서는 더 안정적인 결과를 제공

---

### 배포 및 추론

**모델 다운로드**: [🤗 pathcosmos/EVAFRILL-Mo-3B](https://huggingface.co/pathcosmos/EVAFRILL-Mo-3B)

HF Hub에 **7개 모델 버전** + LoRA 가중치 + preference 데이터 + 학습 설정/스크립트가 모두 업로드되어 있습니다:

| 디렉토리 | 모델 | 설명 |
|----------|------|------|
| `slerp/` | ⭐ **최종 권장** | SFT + DPO SLERP 병합 (α=0.5) |
| `pretrain/` | Pretrain | 319K steps, 55B tokens |
| `sft-v2/` | SFT v2 | 65K steps, val_loss 1.79 |
| `dpo-r1/` | DPO Round 1 | loss 0.693→0.565 |
| `dpo-r2/` | DPO Round 2 | 보수적 미세 조정 |
| `orpo/` | ORPO (실험) | SFT+정렬 동시 학습 |
| `dpo-r3/` | DPO Round 3 (실험) | 반복 특화 |
| `data/` | 재현 데이터 | preference 684K + 반복 특화 105개 |
| `configs/` | 학습 설정 | SFT/DPO/ORPO YAML |
| `scripts/` | 학습 코드 | dpo.py, orpo_native.py, lora.py 등 |

**Gradio 데모 서버:**
```bash
python3 demo/app.py  # http://localhost:7860
```

**GGUF/Ollama/vLLM 지원 현황:**

| 구분 | 지원 여부 | 사유 |
|------|----------|------|
| **llama.cpp/GGUF** | ❌ 불가 | 순수 Mamba-2만 실험적 지원(CPU only), 하이브리드 미지원 |
| **Ollama** | ❌ 불가 | llama.cpp 기반이므로 동일 제약 |
| **vLLM** | ⚠️ 이론상 가능 | Mamba2ForCausalLM 지원하나, 커스텀 가중치 키 매핑 필요 (수일 작업) |
| **Gradio (순수 Python)** | ✅ 작동 중 | `demo/app.py` |

커스텀 하이브리드 아키텍처의 트레이드오프: SSM 상태(Mamba)와 KV 캐시(Attention)를 동시에 관리하는 표준이 GGUF에 없으며, `mamba_ssm` CUDA 커널이 llama.cpp에 미구현. NVIDIA Nemotron-H도 동일 문제 ([llama.cpp #20570](https://github.com/ggml-org/llama.cpp/issues/20570)).

---

### 반복 특화 DPO 실험 (DPO Round 3, 2026-03-25)

#### 실험 동기

기존 DPO는 일반 preference 데이터(504K)로 학습했지만, 반복 문제를 직접 해결하지 못했음 (SFT 79.8% → DPO 80.7%). **반복/비반복 쌍을 명시적으로 포함**하면 DPO가 반복을 직접 타겟할 수 있는지 검증.

#### Self-Generated Preference 데이터

SLERP 모델로 동일 프롬프트에 대해 두 가지 디코딩으로 생성:
- **rejected**: greedy (temp=0, rep_penalty=1.0) → 반복 심함 (평균 71.7%)
- **chosen**: sampling (temp=0.7, rep_penalty=1.2) → 깨끗함 (평균 0.1%)

`data/generate_repetition_preference.py`로 105개 한국어 프롬프트(10개 카테고리: 일상, 과학, 역사, 직업, 건강, 창작, 기술, 문화, 환경 등)에서 105개 preference pairs 생성. 기존 504K와 합쳐 684,647개로 DPO Round 3 실행.

#### 학습 설정 및 결과

| 항목 | 값 |
|------|-----|
| 시작점 | `checkpoints/3b_dpo/checkpoint-slerp` (SLERP 최종 모델) |
| 데이터 | 684,647 pairs (504K 기존 + 105 반복 특화) |
| Steps | 1,000 |
| Beta | 0.05 |
| LR | 1e-7 |
| VRAM | 6.3GB |
| 소요 시간 | ~1.5시간 |

```
학습 추이:
  step   10 | loss 0.6932 | margin -0.007
  step  100 | loss 0.6888 | margin +0.013
  step  500 | loss 0.6925 | margin +0.014
  step 1000 | loss 0.6910 | margin +0.014  (최종)
```

Loss 변화 미미 (0.693→0.691). 이미 SLERP로 잘 정렬된 모델 위에 추가 학습이라 변화폭이 작음. 105개 반복 특화 샘플이 684K 속에 희석됨 (0.015%).

체크포인트: `checkpoints/3b_dpo_r3/checkpoint-merged`

#### 평가 결과

**Greedy 반복률 비교 (15 프롬프트 평균):**

| 모델 | Greedy 반복률 | rep_penalty=1.2 (5p) |
|------|:------------:|:-------------------:|
| SLERP (α=0.5) | **74.5%** | **5.8%** |
| DPO R3 (반복 특화) | 79.4% | 4.5% |

**프롬프트별 상세 (greedy + rep_penalty=1.2):**

| 프롬프트 | SLERP r1.2 | R3 r1.2 |
|----------|:----------:|:-------:|
| 대한민국의 수도는 | 13.4% | **0.4%** |
| 인공지능이란 | **13.4%** | 13.8% |
| 한국의 전통 음식 | 0.0% | 0.0% |
| 건강한 식습관 | **0.8%** | 7.5% |
| 프로그래밍을 배우려면 | 1.6% | **0.8%** |

#### 분석 및 결론

**DPO R3는 SLERP 대비 유의미한 개선을 보이지 않음.**

- Greedy 반복률: SLERP 74.5% → R3 79.4% (**오히려 악화**)
- rep_penalty=1.2: SLERP 5.8% → R3 4.5% (미미한 개선)
- **원인**: 반복 특화 105개가 684K의 0.015%로 희석, 모델 행동에 영향 미미
- **교훈**: self-generated preference 데이터는 최소 수천~수만 개 규모가 필요. 100개 수준으로는 684K 기존 데이터에 묻힘

---

### 향후 개선 방향

1. ~~반복 특화 preference 데이터~~ → ✅ 실험 완료 (위 참조)
2. **반복 특화 데이터 비율 증대** — 105개가 아닌 수천~수만 개로 확대하여 DPO 재학습
3. **SFT 데이터 품질 점검** — hallucination 및 비정상 출력 원인 조사
4. **모델 규모 확대** — 더 큰 compute budget 확보 시 7B+ 규모로 스케일업

---

## 부록: 실행 가이드

### DPO/ORPO 파이프라인 실행

```bash
# DPO Round 1 + Round 2 + SLERP Merge 전체 파이프라인
bash train_3b_dpo_1gpu.sh

# DPO 개별 실행
python3 train/dpo.py \
    --sft_checkpoint checkpoints/3b_sft_v2/checkpoint-best \
    --dpo_data data/preference/combined_preference.jsonl \
    --config configs/h100_mig/dpo_3b_1gpu.yaml \
    --device cuda:0

# SLERP 체크포인트 병합
python3 scripts/merge_checkpoints.py \
    --ckpt_a checkpoints/3b_sft_v2/checkpoint-best \
    --ckpt_b checkpoints/3b_dpo_r1/checkpoint-merged \
    --output checkpoints/3b_dpo/checkpoint-slerp \
    --alpha 0.5
```

### 로그 모니터링

```bash
# DPO 학습 step별 loss/margin/lr
tail -f /root/taketimes/llm/EVAFRILL-Mo/checkpoints/3b_dpo_r1/train.log

# 전체 stdout (모델 로딩, 데이터 파싱 포함)
tail -f /root/taketimes/llm/EVAFRILL-Mo/checkpoints/3b_dpo_r1/stdout.log
```

### 버그 수정 이력

- **LoRA device mismatch** (`model/lora.py`): `lora_A`/`lora_B`가 CPU에 생성되어 GPU 레이어와 device 불일치. `original.weight.device`/`dtype`을 사용하도록 수정.
- **nayohan preference 파서** (`data/prepare_preference_combined.py`): `orig_response_A/B + orig_preference` 형식 지원 추가 (기존 0건 파싱).

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

## 관련 프로젝트

- **[FRANKENSTALLM](https://github.com/pathcosmos/FRANKENSTALLM)** | [🤗 HuggingFace](https://huggingface.co/pathcosmos/frankenstallm) — EVAFRILL-Mo의 전신. 순수 Transformer decoder-only LLM으로 시작한 프로젝트. 한국어+영어+코드+수학 커스텀 토크나이저, DDP 학습 파이프라인 등 기반 인프라를 구축한 프로젝트입니다. EVAFRILL-Mo는 여기서 하이브리드 Mamba-2 + Transformer 아키텍처로 진화했습니다.

NVIDIA Nemotron-H 아키텍처에서 영감을 받아 밑바닥부터 직접 구현한 3B 하이브리드 모델입니다. FRANKENSTALLM이 순수 Transformer 기반이라면, EVAFRILL-Mo는 Mamba-2 SSM + 희소 Transformer 어텐션 하이브리드 구조를 채택했습니다.

| 항목 | FRANKENSTALLM | EVAFRILL-Mo |
|------|---------------|-------------|
| 아키텍처 | 순수 Transformer (28L) | Mamba-2 24L + Attention 2L |
| 파라미터 | 3.17B | 2.94B |
| 핵심 기술 | GQA, FP8, FlashAttention-2 | Selective Scan, SwiGLU FFN in Mamba, GQA |
| 설계 원칙 | 검증된 Transformer 아키텍처 | Nemotron-H 단편화 도입 |
| GPU | 8× B200 | 7× B200 |
| 학습 전략 | Chinchilla-optimal | Chinchilla 93% 달성 목표 |

두 프로젝트는 동일한 토크나이저(64K SentencePiece), 학습 데이터 파이프라인, DDP/FP8 인프라를 공유합니다. "같은 재료, 다른 레시피"로 아키텍처 차이가 성능에 미치는 영향을 비교 실험할 수 있습니다.

---

## 참조 논문

| 논문 | 저자 | 핵심 기여 |
|------|------|-----------|
| [Nemotron-H](https://arxiv.org/abs/2504.03624) | NVIDIA, 2025 | 하이브리드 Mamba-Transformer 아키텍처 설계 |
| [Mamba-2: Structured State Space Duality](https://arxiv.org/abs/2405.21060) | Dao & Gu, 2024 | SSD (Structured State Space Duality) 알고리즘 |
| [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) | Gu & Dao, 2023 | Selective State Space Model 원본 |
| [Chinchilla Scaling Law](https://arxiv.org/abs/2203.15556) | Hoffmann et al., 2022 | 최적 compute 배분 — tokens = 20× params |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Tri Dao, 2023 | IO-aware 어텐션, O(N) 메모리 |
| [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245) | Ainslie et al., 2023 | KV 캐시 효율적 어텐션 |
| [SwiGLU Activation](https://arxiv.org/abs/2002.05202) | Shazeer, 2020 | 게이트 활성화 함수 |
| [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | Su et al., 2021 | 상대적 위치 인코딩 |
| [Scaling Data-Constrained LMs](https://arxiv.org/abs/2305.16264) | Muennighoff et al., 2023 | 데이터 반복 학습의 효과 (최대 4 에포크) |
| [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | Rafailov et al., 2023 | 보상 모델 없는 선호도 정렬 |
| [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691) | Hong et al., 2024 | SFT + 선호도 최적화 단일 단계 통합 |
| [NEFTune](https://arxiv.org/abs/2310.05914) | Jain et al., 2023 | 임베딩 노이즈 주입으로 미세조정 품질 향상 |

---

## 감사의 글

이 프로젝트는 **과학기술정보통신부**의 **「첨단 GPU 활용 지원 사업」** (과학기술정보통신부 공고 제2025-1068호)을 통해 제공된 GPU 컴퓨팅 자원을 활용하여 수행되었습니다.

> **국가 AI컴퓨팅자원 지원포털**: [https://aiinfrahub.kr](https://aiinfrahub.kr)
>
> - 주관: 과학기술정보통신부 (MSIT), 정보통신산업진흥원 (NIPA)
> - 운영: 한국정보통신진흥협회 (KAIT)

대한민국 정부의 AI 인프라 지원 사업 덕분에 7× NVIDIA B200 GPU 환경에서 한국어 3B 하이브리드 Mamba-Transformer 모델을 처음부터 학습할 수 있었습니다. 국가 차원의 AI 컴퓨팅 자원 지원에 깊이 감사드립니다.

- **NVIDIA Nemotron-H** — 하이브리드 Mamba-Transformer 아키텍처 설계의 영감
- **Mamba-2** (Dao & Gu, 2024) — 구조화된 상태 공간 모델의 기반
- **Chinchilla 스케일링 법칙** (Hoffmann et al., 2022) — 최적 학습 compute 배분 기준
- **사용 기술:** PyTorch, FlashAttention-2, TransformerEngine
- **[FRANKENSTALLM](https://github.com/pathcosmos/FRANKENSTALLM)** — 기반 프로젝트

---

## 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다. 상세 내용은 [LICENSE](LICENSE)를 참조하세요.

---

<div align="center">

*EVAFRILL-Mo — 밑바닥부터, selective scan 하나하나 직접 쌓아올린 3B 하이브리드 모델.*

</div>

> 한국어 | **[English](README.en.md)**
