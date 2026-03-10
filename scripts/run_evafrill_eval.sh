#!/bin/bash
# EVAFRILL-Mo 3B 종합 평가 실행
# Usage: bash scripts/run_evafrill_eval.sh [--skip-phase4]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${PROJECT_ROOT}/checkpoints/3b_final/checkpoint-0319772"
LOG_DIR="${PROJECT_ROOT}/checkpoints/3b_final"
LOG_FILE="${LOG_DIR}/eval.log"

cd "${PROJECT_ROOT}"

echo "================================================================"
echo "  EVAFRILL-Mo 3B 종합 평가"
echo "  체크포인트: ${CHECKPOINT}"
echo "  시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

PYTHONPATH="${PROJECT_ROOT}" \
    python eval/evafrill_eval.py \
    --checkpoint "${CHECKPOINT}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "================================================================"
echo "  완료: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Exit code: ${EXIT_CODE}"
echo "================================================================"
