# EVAFRILL-Mo 3B SFT 준비

EVAFRILL-Mo 3B 기준으로 SFT(지도 미세조정)를 진행하기 위한 체크리스트와 실행 방법입니다.

## 1. 사전 확인 (이미 갖춰진 것)

| 항목 | 경로 | 상태 |
|------|------|------|
| Base 체크포인트 | `checkpoints/3b_final/checkpoint-0319772` | Pretrain 완료(319,772 steps) |
| SFT 학습 데이터 | `data/sft_combined/train_filtered.jsonl` | 2,439,397 samples |
| SFT 검증 데이터 | `data/sft_combined/val_filtered.jsonl` | 49,801 samples |
| 토크나이저 | `tokenizer/korean_sp/tokenizer.json` | 사용 |
| SFT 설정 | `configs/korean_3b_sft.yaml` | lr=1e-5, NEFTune 5.0, ~44k steps |
| SFT 스크립트 | `train/sft.py` | — |

## 2. 실행 방법

### 옵션 A: Resilient 래퍼 (권장 — 7 GPU, 재시작 자동)

크래시 시 자동으로 최신 체크포인트부터 재시작합니다.

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-star
bash train_3b_sft_resilient.sh
```

- Base: `checkpoints/3b_final/checkpoint-0319772`
- 출력: `checkpoints/3b_sft/`
- GPU: 7× B200, eff_batch=56, ~44,000 steps (약 1 epoch)

### 옵션 B: launch 스크립트 (7 GPU, 한 번 실행)

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-star
bash scripts/launch_3b_sft.sh
```

- Base: `checkpoints/3b_final/checkpoint-0319772` (기본값)
- 출력: `checkpoints/3b_sft/train.log` 등
- 재시작: `bash scripts/launch_3b_sft.sh --resume checkpoints/3b_sft/checkpoint-XXXXX`

### 빠른 동작 확인 (2 step만)

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-star
bash scripts/launch_3b_sft.sh --max_steps 2
```

## 3. 데이터를 아직 안 만들었다면

SFT 데이터가 없을 때만 순서대로 실행:

```bash
cd /PROJECT/0325120031_A/ghong/taketimes/llm-star
# 1) 통합 (llm-bang/data/sft_* 또는 동일 구조 필요)
bash scripts/prepare_sft_combined.sh
# 2) 품질 필터
python data/filter_sft_v2.py \
  --input  data/sft_combined/train.jsonl \
  --output data/sft_combined/train_filtered.jsonl
```

## 4. 설정 요약 (korean_3b_sft.yaml)

- **max_steps**: 44,000 (≈1 epoch)
- **batch_size**: 2 per GPU, **grad_accum**: 4 → eff_batch 56 (7 GPU)
- **lr**: 1e-5, **warmup_steps**: 500
- **neftune_alpha**: 5.0 (반복 완화)
- **save_interval**: 5,000, **eval_interval**: 1,000

## 5. 모니터링

```bash
tail -f checkpoints/3b_sft/train.log
# 또는 resilient 사용 시
tail -f checkpoints/3b_sft/monitor.log
```

---

**요약**: Base·데이터·설정은 준비된 상태입니다. `bash train_3b_sft_resilient.sh` 또는 `bash scripts/launch_3b_sft.sh` 로 SFT를 시작하면 됩니다.
