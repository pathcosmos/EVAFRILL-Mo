# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 작업 원칙 — 팀플레이

**병렬 처리 가능한 작업은 항상 서브 에이전트로 분배한다.**

- 복잡한 코드 작성 / 설계 판단 → `model: sonnet`
- 빠른 탐색 · 조회 · 간단한 파일 작성 → `model: haiku`
- 에이전트 완료 후 결과 회수; 필요 시 `resume` 으로 재호출
- 예: 모델 구현(sonnet) + 데이터 스크립트(sonnet) + 설정 파일(haiku) 동시 실행

---

## 프로젝트 목적

소규모 LLM(Large Language Model) 실험 프로젝트.
8× NVIDIA B200 GPU 환경에서 LLM **사전학습(pretraining)** 또는 **파인튜닝(fine-tuning)** 을 직접 구현하고 실험한다.

---

## 하드웨어 환경

| 항목 | 사양 |
|------|------|
| GPU | 8× NVIDIA B200 (183 GB VRAM each, **~1.47 TB total**) |
| RAM | 2.2 TB |
| CUDA | 13.0 |
| Storage (작업) | `/PROJECT/0325120031_A/ghong/taketimes/` → 3.5 TB, 여유 2.2 TB |
| Storage (홈) | `/home/ghong` → 5 GB (소규모 코드만 저장) |

**주의**: 체크포인트, 데이터셋 등 대용량 파일은 반드시 `/PROJECT/0325120031_A/ghong/taketimes/llm-bang/` 하위에 저장할 것. 홈 디렉토리(`/home/ghong`) 용량 초과 주의.

---

## 사전 설치된 라이브러리

```
torch          2.10.0a0+b4e4ee81d3.nv25.12   # NV 커스텀 빌드 (B200 최적화)
flash_attn     2.7.4.post1+25.12              # FlashAttention-2 사용 가능
datasets       4.4.1
tokenizers     0.22.1
huggingface_hub 1.2.3
```

> **경고**: PyTorch는 NVIDIA 커스텀 빌드(`nv25.12`)가 설치됨. `pip install torch` 로 재설치하면 B200 최적화가 깨질 수 있음 — PyTorch 재설치 금지.

## 추가 설치 필요 라이브러리

```bash
pip install transformers accelerate peft trl deepspeed bitsandbytes sentencepiece wandb
```

---

## 권장 프로젝트 구조

```
llm-bang/
├── CLAUDE.md
├── data/               # 학습 데이터 (원본 텍스트, 전처리 완료본)
├── tokenizer/          # 토크나이저 학습·저장
├── model/              # 모델 아키텍처 정의 (nn.Module)
├── train/              # 학습 스크립트 (단일 GPU / DDP / FSDP)
├── eval/               # 평가 스크립트 (perplexity, downstream task)
├── configs/            # YAML/JSON 학습 설정 파일
└── checkpoints/        # 모델 체크포인트 (대용량)
```

---

## 멀티-GPU 학습 실행 패턴

```bash
# torchrun (DDP) — 8 GPU
torchrun --nproc_per_node=8 train/pretrain.py --config configs/small_lm.yaml

# 단일 GPU 테스트
python train/pretrain.py --config configs/small_lm.yaml --device cuda:0

# FSDP (모델 샤딩, 대형 모델)
torchrun --nproc_per_node=8 train/pretrain.py --config configs/large_lm.yaml --strategy fsdp
```

---

## 모델 규모 가이드 (하드웨어 기준)

| 모델 크기 | 추천 전략 | 최소 GPU 수 |
|-----------|-----------|------------|
| ~1B param | DDP, bf16 | 1 GPU |
| ~7B param | DDP 또는 FSDP, bf16 | 2–4 GPU |
| ~13B param | FSDP, bf16/fp8 | 4 GPU |
| ~70B param | FSDP + ZeRO-3, bf16/fp8 | 8 GPU |

B200은 FP8 네이티브 지원 → 학습 시 `torch.float8_e4m3fn` 활용 가능.

---

## 참고 (이전 프로젝트)

`/PROJECT/0325120031_A/ghong/taketimes/_deprecated/work/` — 2CRM 두께 실측값 예측(LightGBM, ClickHouse) 프로젝트.
도메인 데이터(공장 센서, 코일 그레이드) 필요 시 참고.
