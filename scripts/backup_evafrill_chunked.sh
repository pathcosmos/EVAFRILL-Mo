#!/usr/bin/env bash
# EVAFRILL-Mo 한정 백업: 3B 체크포인트, 평가 스크립트/결과, 리포트, 모델 소스, 토크나이저, 설정
# 990MB 단위로 분할 압축 → llm-star-backup/<날짜>/evafrill-mo.tar.zst.*
# 사용: bash scripts/backup_evafrill_chunked.sh [날짜폴더명]

set -euo pipefail

PROJECT_ROOT="/PROJECT/0325120031_A/ghong/taketimes"
LLM_STAR="${PROJECT_ROOT}/llm-star"
BACKUP_BASE="${PROJECT_ROOT}/llm-star-backup"
DATE_FOLDER="${1:-$(date +%Y%m%d)}"
BACKUP_DIR="${BACKUP_BASE}/${DATE_FOLDER}"
CHUNK_SIZE="990M"
ARCHIVE_NAME="evafrill-mo"

mkdir -p "${BACKUP_DIR}"
cd "${PROJECT_ROOT}"

echo "[backup] EVAFRILL-Mo only → ${BACKUP_DIR} (chunks ${CHUNK_SIZE})"
echo "[backup] Including: checkpoints/3b_final, model, tokenizer/korean_sp, eval (evafrill), scripts, reports, configs, README, CLAUDE.md"

tar cf - \
  -C "${PROJECT_ROOT}" \
  llm-star/checkpoints/3b_final \
  llm-star/model \
  llm-star/tokenizer/korean_sp \
  llm-star/eval/evafrill_eval.py \
  llm-star/eval/outputs \
  llm-star/scripts/run_evafrill_eval.sh \
  llm-star/reports \
  llm-star/configs \
  llm-star/README.md \
  llm-star/CLAUDE.md \
  | zstd -3 -T0 \
  | split -b "${CHUNK_SIZE}" - "${BACKUP_DIR}/${ARCHIVE_NAME}.tar.zst."

echo "[backup] Writing MANIFEST and SHA256SUMS..."
echo "EVAFRILL-Mo backup date=${DATE_FOLDER} chunks=${CHUNK_SIZE}" > "${BACKUP_DIR}/MANIFEST_evafrill.txt"
ls -la "${BACKUP_DIR}"/${ARCHIVE_NAME}.tar.zst.* 2>/dev/null >> "${BACKUP_DIR}/MANIFEST_evafrill.txt" || true
(cd "${BACKUP_DIR}" && sha256sum ${ARCHIVE_NAME}.tar.zst.* > SHA256SUMS_evafrill 2>/dev/null || true)
echo "[backup] Done. Restore: cat ${ARCHIVE_NAME}.tar.zst.* | zstd -d | tar xvf - -C ${PROJECT_ROOT}"
