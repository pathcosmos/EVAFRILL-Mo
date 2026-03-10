# EVAFRILL-Mo 3B SFT — 클라우드 이전 및 비용 분석

> 작성일: 2026-03-10
> 모델: EVAFRILL-Mo 3B (Hybrid Mamba-2 + Transformer, 2.95B params)

---

## 1. 현재 SFT 진행 상태

| 항목 | 값 |
|------|-----|
| Base checkpoint | `checkpoints/3b_final/checkpoint-0319772` (pretrain 319,772 steps) |
| SFT 데이터 | 3,774,413 samples (train), 77,207 samples (val) |
| 설정 | lr=1e-5, eff_batch=56, NEFTune α=5.0, FP8 |
| 진행 | step ~11,000 / 44,000 (25%) |
| 장비 | 7× NVIDIA B200 (183GB each) |
| 속도 | ~0.5초/step, ~120 steps/분 |

### 저장된 체크포인트

| 체크포인트 | 설명 |
|-----------|------|
| `checkpoints/3b_sft/checkpoint-0005000` | 정기 저장 (step 5,000) |
| `checkpoints/3b_sft/checkpoint-0010000` | 정기 저장 (step 10,000) |
| `checkpoints/3b_sft/checkpoint-best` | val loss 최저점 |

---

## 2. VRAM 요구량 분석

### 3B 모델 메모리 상세

| 컴포넌트 | 계산 | 크기 |
|----------|------|------|
| Model weights (BF16) | 2.95B × 2 bytes | 5.9 GB |
| AdamW optimizer (FP32 m+v) | 2.95B × 8 bytes | 23.6 GB |
| Gradients (BF16) | 2.95B × 2 bytes | 5.9 GB |
| FP32 master copy | 2.95B × 4 bytes | 11.8 GB |
| Activations (with checkpointing) | batch=2, seq=4096 | 3-5 GB |
| CUDA 커널/버퍼 | | 1-2 GB |
| **합계 (Full SFT)** | | **52-56 GB** |

### 추론 시 메모리

| 모드 | VRAM |
|------|------|
| BF16 추론 | 7-8 GB |
| INT8 양자화 | 4-5 GB |
| INT4 양자화 (GPTQ/AWQ) | 2.5-3.5 GB |

---

## 3. 장비별 가능 작업

| 작업 | 5070 Ti (16GB) | A100 40GB | H100 80GB |
|------|:-:|:-:|:-:|
| Full SFT | **불가** | **불가** (52GB) | **가능** |
| QLoRA SFT | 가능 (느림) | **가능** | **가능** |
| Full ORPO/DPO | **불가** | **불가** | **가능** |
| ORPO QLoRA | 힘듦 | **가능** | **가능** |
| 추론 BF16 | **가능** (8GB) | **가능** | **가능** |
| 추론 INT4 | **여유** (3GB) | **여유** | **여유** |
| 평가 (eval) | **가능** | **가능** | **가능** |
| 양자화 변환 | **가능** | **가능** | **가능** |

### 로컬 장비 활용 전략

- **5070 Ti (16GB)**: 평가, INT4 추론/서빙, 양자화 변환에 활용
- **A100 40GB**: QLoRA SFT, QLoRA ORPO, 평가, 추론 모두 가능
- Full SFT/ORPO는 H100 80GB (또는 multi-GPU) 필수

---

## 4. 클라우드 비용 분석

### GPU 시간당 가격 (2026년 기준, on-demand)

| GPU | Vast.ai | RunPod | Lambda | CoreWeave |
|-----|---------|--------|--------|-----------|
| H100 80GB | $1.49-1.98 | $1.99-2.39 | $2.99 | $6.16 |
| A100 80GB | $1.15-1.57 | $1.39-1.49 | $2.06 | — |
| A100 40GB | ~$1.15 | ~$1.20 | — | — |
| 4× H100 node | — | $8-10 | ~$12 | ~$25 |

### 작업별 비용 예측

| 작업 | GPU 구성 | 시간 | 단가/hr | 비용 |
|------|----------|------|---------|------|
| SFT 1라운드 잔여 (~33K steps) | 4× H100 | ~10h | $8-10 | **$80-100** |
| SFT 2라운드 (44K steps, 1 epoch) | 4× H100 | ~12h | $8-10 | **$96-120** |
| ORPO (~10-20K steps) | 4× H100 | ~6h | $8-10 | **$48-60** |
| 평가 (벤치마크 suite) | 1× H100 | ~2h | $2-3 | **$4-6** |
| 양자화 + 배포 준비 | 1× H100 | ~2h | $2-3 | **$4-6** |

### 총 예산

| 시나리오 | 범위 | 비고 |
|----------|------|------|
| **최소** (SFT잔여 + ORPO + eval) | **$130-170** | RunPod/Vast.ai 기준 |
| **표준** (SFT 2라운드 + ORPO + eval + 배포) | **$230-300** | 시행착오 약간 포함 |
| **여유** (실험 반복 + 디버깅 포함) | **$400-500** | Lambda/안정적 플랫폼 |

---

## 5. 권장 워크플로우

```
Phase 1: SFT 1라운드 완료
├─ 현재 B200에서 step 11,000까지 완료
├─ 클라우드 이전 시 checkpoint-best 또는 checkpoint-0010000에서 --resume
└─ 남은 ~33K steps: 4×H100에서 ~10시간

Phase 2: 1차 평가
├─ 로컬 5070Ti에서 가능 (INT4 추론)
├─ MMLU-ko, 생성 품질, 반복율 측정
└─ eval/evafrill_eval.py 사용

Phase 3: SFT 2라운드 (선택)
├─ 추가 데이터 또는 동일 데이터 2 epoch
└─ 클라우드 H100, ~12시간

Phase 4: ORPO 정렬 학습
├─ preference 데이터 준비 필요 (chosen/rejected pairs)
├─ ORPO는 reference model 불필요 → DPO 대비 VRAM 30% 절약
└─ 클라우드 H100, ~6시간

Phase 5: 최종 평가 + 양자화
├─ 로컬 5070Ti에서 모두 가능
├─ GPTQ/AWQ INT4 양자화
└─ 서빙 테스트

Phase 6: 배포
├─ 5070Ti (16GB)로 INT4 서빙 가능
├─ vLLM 또는 llama.cpp 기반
└─ 3B INT4 ≈ 3GB VRAM → 여유
```

---

## 6. 클라우드 이전 시 필요 파일

```
# 필수 (SFT 이어가기)
checkpoints/3b_sft/checkpoint-best/       # 17GB (model+optimizer+scheduler+state)
checkpoints/3b_final/checkpoint-0319772/  # base checkpoint (SFT 코드가 참조)
configs/korean_3b_sft.yaml
train/sft.py
train/trainer.py
train/utils.py
data/sft_dataset.py
model/                                     # 전체 모델 코드
tokenizer/korean_sp/tokenizer.json

# 데이터
data/sft_combined/train_filtered.jsonl    # 7.5GB
data/sft_combined/val_filtered.jsonl

# 실행 스크립트
scripts/launch_3b_sft.sh
train_3b_sft_resilient.sh
```

### 이전 후 재개 명령

```bash
# 클라우드 H100 환경에서
pip install torch transformers tokenizers sentencepiece

# GPU 수에 맞게 grad_accum 조정 (eff_batch=56 유지)
# 4× H100: batch=2, grad_accum=7 → eff_batch = 2×4×7 = 56
torchrun --nproc_per_node=4 train/sft.py \
    --config configs/korean_3b_sft.yaml \
    --base_checkpoint checkpoints/3b_final/checkpoint-0319772 \
    --sft_data data/sft_combined/train_filtered.jsonl \
    --val_data data/sft_combined/val_filtered.jsonl \
    --checkpoint_dir checkpoints/3b_sft \
    --resume checkpoints/3b_sft/checkpoint-best \
    --grad_accum 7 \
    --use_fp8 \
    --seed 42
```

---

## 7. 비용 절감 팁

1. **Spot/Preemptible 인스턴스**: 50-70% 저렴, resilient wrapper가 자동 재시작 지원
2. **Vast.ai Community**: H100 $1.49/hr로 최저가
3. **QLoRA 대안**: A100 40GB($1.15/hr)에서 QLoRA SFT → 비용 60% 절감, 품질은 약간 하락
4. **평가/양자화는 로컬**: 5070Ti에서 무료로 수행
5. **ORPO > DPO**: reference model 불필요 → 같은 GPU에서 더 큰 batch 가능
