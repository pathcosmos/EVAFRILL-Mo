#!/usr/bin/env bash
# EVAFRILL-Mo 3B SFT 사전 확인 — 필수 경로·데이터 검사 후 실행 안내 출력
# 사용: bash scripts/prepare_evafrill_sft.sh

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_CKPT="checkpoints/3b_final/checkpoint-0319772"
TRAIN_DATA="data/sft_combined/train_filtered.jsonl"
VAL_DATA="data/sft_combined/val_filtered.jsonl"
TOKENIZER="tokenizer/korean_sp/tokenizer.json"

echo "=========================================="
echo "  EVAFRILL-Mo 3B SFT 준비 확인"
echo "=========================================="

OK=0
MISS=0

if [[ -d "${BASE_CKPT}" ]]; then
  echo "  [OK] Base checkpoint: ${BASE_CKPT}"
  ((OK++)) || true
else
  echo "  [MISS] Base checkpoint: ${BASE_CKPT}"
  ((MISS++)) || true
fi

if [[ -f "${TRAIN_DATA}" ]]; then
  n=$(wc -l < "${TRAIN_DATA}")
  echo "  [OK] SFT train: ${TRAIN_DATA} (${n} lines)"
  ((OK++)) || true
else
  echo "  [MISS] SFT train: ${TRAIN_DATA}"
  echo "         → bash scripts/prepare_sft_combined.sh && python data/filter_sft_v2.py --input data/sft_combined/train.jsonl --output ${TRAIN_DATA}"
  ((MISS++)) || true
fi

if [[ -f "${VAL_DATA}" ]]; then
  n=$(wc -l < "${VAL_DATA}")
  echo "  [OK] SFT val: ${VAL_DATA} (${n} lines)"
  ((OK++)) || true
else
  echo "  [MISS] SFT val: ${VAL_DATA}"
  ((MISS++)) || true
fi

if [[ -f "${TOKENIZER}" ]]; then
  echo "  [OK] Tokenizer: ${TOKENIZER}"
  ((OK++)) || true
else
  echo "  [MISS] Tokenizer: ${TOKENIZER}"
  ((MISS++)) || true
fi

echo "------------------------------------------"
if [[ $MISS -gt 0 ]]; then
  echo "  일부 항목이 없습니다. 위 [MISS] 안내대로 준비한 뒤 다시 실행하세요."
  exit 1
fi

echo "  준비 완료. SFT 시작 방법:"
echo ""
echo "  # Resilient (7 GPU, 자동 재시작)"
echo "  bash train_3b_sft_resilient.sh"
echo ""
echo "  # 또는 launch (7 GPU)"
echo "  bash scripts/launch_3b_sft.sh"
echo ""
echo "  # 동작 확인만 (2 step)"
echo "  bash scripts/launch_3b_sft.sh --max_steps 2"
echo "=========================================="
