#!/usr/bin/env bash
# data/download_cc100.sh
# CC-100 Korean 데이터 단독 다운로드 스크립트
#
# 버그 수정 내역 (build_korean_dataset.sh 대비):
#   - cc100 데이터셋의 텍스트 컬럼명은 'text'가 아닌 'sentence' 임.
#     build_korean_dataset.sh Step 1에서 --text_col text 로 잘못 지정되어
#     모든 행이 빈 문자열로 처리되는 버그가 있었음.
#     본 스크립트는 --text_col sentence 로 올바르게 지정한다.
#
# 실행 방법 (프로젝트 루트에서):
#   bash data/download_cc100.sh
#
# 출력:
#   data/raw/cc100_ko/cc100_train_XXXX.txt  (100,000행 단위 샤드)
#
# 주의:
#   - cc100_ko 디렉토리에 이미 .txt 파일이 있으면 다운로드를 건너뜀.
#   - 대용량 파일은 /PROJECT/0325120031_A/ghong/taketimes/ 하위에만 저장할 것.

set -euo pipefail

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RAW_DIR="data/raw"
CC100_DIR="$RAW_DIR/cc100_ko"

# ─── 다운로드 파라미터 ────────────────────────────────────────────────────────
CC100_MAX_ROWS=10000000   # 1,000만 행 (~1.5B 토큰 추정)
CC100_SHARD_SIZE=100000   # 샤드 당 행 수
CC100_TEXT_COL="sentence" # cc100 데이터셋의 실제 텍스트 컬럼명 (text 아님!)

# ─── 이미 완료된 경우 건너뜀 ─────────────────────────────────────────────────
echo "=== CC-100 Korean 다운로드 ==="
echo "프로젝트 루트: $PROJECT_ROOT"
echo "출력 디렉토리: $CC100_DIR"
echo ""

mkdir -p "$CC100_DIR"

# cc100_ko 디렉토리에 .txt 파일이 하나라도 있으면 스킵
EXISTING_COUNT=$(find "$CC100_DIR" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l)
if [ "$EXISTING_COUNT" -gt 0 ]; then
    echo "[SKIP] $CC100_DIR 에 이미 ${EXISTING_COUNT}개 .txt 파일이 존재합니다."
    echo "       재다운로드 하려면 해당 디렉토리를 비운 뒤 다시 실행하세요."
    echo "       rm -f \"$CC100_DIR\"/*.txt"
    exit 0
fi

# ─── CC-100 다운로드 ──────────────────────────────────────────────────────────
echo "[다운로드] CC-100 Korean (max_rows=$CC100_MAX_ROWS, text_col=$CC100_TEXT_COL)..."
echo "  주의: HuggingFace cc100 데이터셋의 텍스트 컬럼명은 'sentence' 입니다."
echo ""

python data/download.py \
    --dataset cc100 \
    --subset ko \
    --split train \
    --text_col "$CC100_TEXT_COL" \
    --output_dir "$CC100_DIR" \
    --shard_size "$CC100_SHARD_SIZE" \
    --max_rows "$CC100_MAX_ROWS"

# ─── 결과 확인 ────────────────────────────────────────────────────────────────
echo ""
FINAL_COUNT=$(find "$CC100_DIR" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l)
if [ "$FINAL_COUNT" -gt 0 ]; then
    TOTAL_BYTES=$(du -sh "$CC100_DIR" 2>/dev/null | cut -f1)
    echo "=== 완료 ==="
    echo "  생성된 샤드 파일: ${FINAL_COUNT}개"
    echo "  디렉토리 총 용량: ${TOTAL_BYTES}"
    echo "  경로: $CC100_DIR"
    echo ""
    echo "다음 단계: CC-100 토크나이징 & 기존 데이터와 병합"
    echo "  bash data/tokenize_cc100.sh"
else
    echo "ERROR: 다운로드 후 .txt 파일이 생성되지 않았습니다." >&2
    echo "  download.py 출력을 확인하고 cc100 데이터셋 접근 가능 여부를 점검하세요." >&2
    exit 1
fi
