#!/usr/bin/env bash
# data/tokenize_cc100.sh
# CC-100 Korean 토크나이징 및 기존 korean_train.bin 과의 병합 스크립트
#
# 버그 수정 내역 (build_korean_dataset.sh 대비):
#   - build_korean_dataset.sh Step 6에서 cc100_ko 디렉토리가 비어있을 경우
#     prepare.py 의 find_input_files()가 FileNotFoundError 를 발생시키는 버그가 있었음.
#     본 스크립트는 사전에 cc100_ko/*.txt 파일 존재 여부를 확인하고
#     없을 경우 명확한 안내 메시지와 함께 종료한다.
#
# 전제 조건:
#   1. tokenizer/korean_sp/tokenizer.json  — SP 토크나이저가 이미 학습/변환 완료
#   2. data/raw/cc100_ko/*.txt             — CC-100 다운로드 완료
#      (없으면: bash data/download_cc100.sh 먼저 실행)
#   3. data/korean_train.bin               — 기존 병합 학습 데이터 (병합 대상)
#      (없어도 토크나이징은 진행되며, 병합 단계만 건너뜀)
#
# 실행 방법 (프로젝트 루트에서):
#   bash data/tokenize_cc100.sh
#
# 출력:
#   data/korean_cc100_train.bin  — CC-100 학습 토큰
#   data/korean_cc100_val.bin    — CC-100 검증 토큰
#   data/korean_train_combined.bin  — 기존 korean_train.bin + CC-100 병합본
#                                     (korean_train.bin 이 존재하는 경우에만 생성)

set -euo pipefail

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RAW_DIR="data/raw"
BIN_DIR="data"
TOKENIZER_JSON="tokenizer/korean_sp/tokenizer.json"
CC100_DIR="$RAW_DIR/cc100_ko"

# ─── 출력 파일 경로 ───────────────────────────────────────────────────────────
CC100_TRAIN_BIN="$BIN_DIR/korean_cc100_train.bin"
CC100_VAL_BIN="$BIN_DIR/korean_cc100_val.bin"
EXISTING_TRAIN_BIN="$BIN_DIR/korean_train.bin"
COMBINED_TRAIN_BIN="$BIN_DIR/korean_train_combined.bin"

# ─── 사전 검사 ────────────────────────────────────────────────────────────────
echo "=== CC-100 토크나이징 및 병합 ==="
echo "프로젝트 루트: $PROJECT_ROOT"
echo ""

# 검사 1: 토크나이저 파일 존재 여부
if [ ! -f "$TOKENIZER_JSON" ]; then
    echo "ERROR: 토크나이저 파일을 찾을 수 없습니다: $TOKENIZER_JSON" >&2
    echo ""
    echo "해결 방법: 토크나이저를 먼저 학습하고 변환하세요."
    echo "  python tokenizer/train_sp_tokenizer.py --input <텍스트파일> --output_dir tokenizer/korean_sp"
    echo "  python tokenizer/convert_sp_to_hf.py --model tokenizer/korean_sp/tokenizer.model --output $TOKENIZER_JSON"
    exit 1
fi
echo "[OK] 토크나이저: $TOKENIZER_JSON"

# 검사 2: CC-100 .txt 파일 존재 여부
CC100_FILE_COUNT=$(find "$CC100_DIR" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l)
if [ "$CC100_FILE_COUNT" -eq 0 ]; then
    echo "ERROR: CC-100 텍스트 파일이 없습니다: $CC100_DIR/*.txt" >&2
    echo ""
    echo "해결 방법: CC-100 먼저 다운로드하세요."
    echo "  bash data/download_cc100.sh"
    echo ""
    echo "주의: build_korean_dataset.sh 의 --text_col text 버그로 다운로드했다면"
    echo "      해당 파일들은 빈 내용이므로 삭제 후 재다운로드가 필요합니다."
    echo "  rm -f \"$CC100_DIR\"/*.txt && bash data/download_cc100.sh"
    exit 1
fi
echo "[OK] CC-100 샤드 파일: ${CC100_FILE_COUNT}개 ($CC100_DIR)"

# 검사 3: 기존 korean_train.bin 존재 여부 확인 (경고만, 중단하지 않음)
if [ -f "$EXISTING_TRAIN_BIN" ]; then
    EXISTING_SIZE=$(du -sh "$EXISTING_TRAIN_BIN" 2>/dev/null | cut -f1)
    echo "[OK] 기존 학습 데이터: $EXISTING_TRAIN_BIN ($EXISTING_SIZE) — 병합 예정"
else
    echo "[WARN] 기존 학습 데이터 없음: $EXISTING_TRAIN_BIN"
    echo "       토크나이징만 진행하고, 병합 단계는 건너뜁니다."
fi

echo ""

# ─── Step 1: CC-100 토크나이징 ────────────────────────────────────────────────
# prepare.py 는 --output 경로의 'train' 을 'val' 로 치환하여 val .bin 을 자동 생성함.
# --val_split 0.002 → 0.2% 를 검증 셋으로 분리 (1,000만 행 기준 약 3M 토큰)

echo "[1/2] CC-100 토크나이징..."
echo "  입력: $CC100_DIR/*.txt  (${CC100_FILE_COUNT}개 파일)"
echo "  출력: $CC100_TRAIN_BIN"
echo "  출력: $CC100_VAL_BIN   (val_split=0.2%)"
echo ""

python data/prepare.py \
    --input "$CC100_DIR/*.txt" \
    --output "$CC100_TRAIN_BIN" \
    --tokenizer "$TOKENIZER_JSON" \
    --val_split 0.002 \
    --seed 42

echo ""
echo "[완료] 토크나이징 결과:"
if [ -f "$CC100_TRAIN_BIN" ]; then
    echo "  $CC100_TRAIN_BIN  ($(du -sh "$CC100_TRAIN_BIN" | cut -f1))"
fi
if [ -f "$CC100_VAL_BIN" ]; then
    echo "  $CC100_VAL_BIN   ($(du -sh "$CC100_VAL_BIN" | cut -f1))"
fi
echo ""

# ─── Step 2: 기존 korean_train.bin 과 병합 ────────────────────────────────────
# 병합 결과는 korean_train_combined.bin 으로 저장.
# 기존 korean_train.bin 은 덮어쓰지 않으므로 안전하게 검토 후 교체 가능.

if [ -f "$EXISTING_TRAIN_BIN" ] && [ -f "$CC100_TRAIN_BIN" ]; then
    echo "[2/2] 기존 학습 데이터와 병합..."
    echo "  입력1: $EXISTING_TRAIN_BIN"
    echo "  입력2: $CC100_TRAIN_BIN"
    echo "  출력:  $COMBINED_TRAIN_BIN"
    echo ""

    python data/merge_bins.py \
        "$EXISTING_TRAIN_BIN" \
        "$CC100_TRAIN_BIN" \
        "$COMBINED_TRAIN_BIN"

    echo ""
    echo "[완료] 병합 결과:"
    echo "  $COMBINED_TRAIN_BIN  ($(du -sh "$COMBINED_TRAIN_BIN" | cut -f1))"
    echo ""
    echo "병합 파일을 기존 학습 데이터로 교체하려면:"
    echo "  mv \"$EXISTING_TRAIN_BIN\" \"${EXISTING_TRAIN_BIN%.bin}_backup.bin\""
    echo "  mv \"$COMBINED_TRAIN_BIN\" \"$EXISTING_TRAIN_BIN\""
else
    echo "[2/2] 병합 건너뜀 — 기존 korean_train.bin 없음."
    echo "  CC-100 학습 데이터만 단독으로 생성되었습니다: $CC100_TRAIN_BIN"
fi

# ─── 최종 요약 ────────────────────────────────────────────────────────────────
echo ""
echo "=== 완료 ==="
echo ""
echo "생성된 파일:"
for f in "$CC100_TRAIN_BIN" "$CC100_VAL_BIN" "$COMBINED_TRAIN_BIN"; do
    if [ -f "$f" ]; then
        TOKEN_COUNT=$(python3 -c "
import numpy as np, sys
d = np.memmap('$f', dtype='uint16', mode='r')
print(f'{len(d):,}')
" 2>/dev/null || echo "계산 불가")
        echo "  $f  →  ${TOKEN_COUNT} 토큰  ($(du -sh "$f" | cut -f1))"
    fi
done
echo ""
echo "학습 재시작 시 combined 파일을 configs/small_fp8_run1.yaml 의"
echo "data_path 에 지정하거나, 기존 korean_train.bin 을 교체하세요."
