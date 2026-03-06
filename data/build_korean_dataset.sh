#!/usr/bin/env bash
# data/build_korean_dataset.sh
# 한국어 LLM 학습 데이터 전체 파이프라인 자동화
#
# 실행 방법:
#   bash data/build_korean_dataset.sh
#
# 단계:
#   1. CC-100 Korean 다운로드
#   2. mC4 Korean 다운로드
#   3. Namuwiki 다운로드
#   4. SentencePiece 토크나이저 학습 (tokenizer/train_sp_tokenizer.py)
#   5. SP → HuggingFace tokenizers.json 변환
#   6. 각 소스 토크나이징 (prepare.py)
#   7. .bin 파일 병합 (merge_bins.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ─── 설정 ─────────────────────────────────────────────────────────────────
RAW_DIR="data/raw"
BIN_DIR="data"
TOKENIZER_DIR="tokenizer/korean_sp"
VOCAB_SIZE=64000

# CC-100: 1,000만 행 (~1.5B 토큰) — 전체는 80M+ 행이므로 먼저 샘플
CC100_MAX_ROWS=10000000
C4_MAX_ROWS=5000000

echo "=== 한국어 LLM 데이터 파이프라인 ==="
echo "작업 디렉토리: $PROJECT_ROOT"
echo ""

# ─── Step 1: CC-100 Korean 다운로드 ──────────────────────────────────────
echo "[1/7] CC-100 Korean 다운로드..."
mkdir -p "$RAW_DIR/cc100_ko"
python data/download.py \
    --dataset cc100 \
    --subset ko \
    --text_col text \
    --output_dir "$RAW_DIR/cc100_ko" \
    --shard_size 100000 \
    --max_rows $CC100_MAX_ROWS
echo ""

# ─── Step 2: mC4 Korean 다운로드 ─────────────────────────────────────────
echo "[2/7] mC4 Korean 다운로드..."
mkdir -p "$RAW_DIR/c4_ko"
python data/download.py \
    --dataset allenai/c4 \
    --subset ko \
    --split train \
    --text_col text \
    --output_dir "$RAW_DIR/c4_ko" \
    --shard_size 100000 \
    --max_rows $C4_MAX_ROWS
echo ""

# ─── Step 3: Namuwiki 다운로드 ───────────────────────────────────────────
echo "[3/7] Namuwiki 다운로드..."
mkdir -p "$RAW_DIR/namuwiki_ko"
python data/download.py \
    --dataset heegyu/namuwiki-extracted \
    --text_col text \
    --output_dir "$RAW_DIR/namuwiki_ko" \
    --shard_size 100000
echo ""

# ─── Step 4: SentencePiece 토크나이저 학습 ──────────────────────────────
echo "[4/7] SentencePiece Unigram 토크나이저 학습 (vocab=$VOCAB_SIZE)..."
mkdir -p "$TOKENIZER_DIR"
# Namuwiki(소형, 빠름) + ko_wiki(기존)를 시드 텍스트로 사용
INPUT_FOR_SP=""
for dir in "$RAW_DIR/namuwiki_ko" "data/raw"; do
    txts=$(find "$dir" -maxdepth 1 -name "*.txt" 2>/dev/null | head -20 | tr '\n' ',')
    INPUT_FOR_SP="${INPUT_FOR_SP}${txts}"
done
INPUT_FOR_SP="${INPUT_FOR_SP%,}"  # trailing comma 제거

python tokenizer/train_sp_tokenizer.py \
    --input "$INPUT_FOR_SP" \
    --vocab_size $VOCAB_SIZE \
    --output_dir "$TOKENIZER_DIR"
echo ""

# ─── Step 5: SP → HF tokenizers.json 변환 ───────────────────────────────
echo "[5/7] SentencePiece → HuggingFace tokenizers.json 변환..."
python tokenizer/convert_sp_to_hf.py \
    --model "$TOKENIZER_DIR/tokenizer.model" \
    --output "$TOKENIZER_DIR/tokenizer.json"
echo ""

# ─── Step 6: 토크나이징 ──────────────────────────────────────────────────
echo "[6/7] 데이터 토크나이징..."

python data/prepare.py \
    --input "$RAW_DIR/cc100_ko/*.txt" \
    --output "$BIN_DIR/korean_cc100_train.bin" \
    --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
    --val_split 0.002 \
    --seed 42

python data/prepare.py \
    --input "$RAW_DIR/c4_ko/*.txt" \
    --output "$BIN_DIR/korean_c4_train.bin" \
    --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
    --val_split 0.002 \
    --seed 43

python data/prepare.py \
    --input "$RAW_DIR/namuwiki_ko/*.txt" \
    --output "$BIN_DIR/korean_namuwiki_train.bin" \
    --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
    --val_split 0.002 \
    --seed 44

echo ""

# ─── Step 7: .bin 병합 ────────────────────────────────────────────────────
echo "[7/7] 학습 데이터 병합..."

# 훈련 셋 병합
TRAIN_BINS=$(ls "$BIN_DIR"/korean_*_train.bin 2>/dev/null | tr '\n' ' ')
if [ -n "$TRAIN_BINS" ]; then
    python data/merge_bins.py $TRAIN_BINS "$BIN_DIR/korean_train.bin"
fi

# 검증 셋 병합
VAL_BINS=$(ls "$BIN_DIR"/korean_*_val.bin 2>/dev/null | tr '\n' ' ')
if [ -n "$VAL_BINS" ]; then
    python data/merge_bins.py $VAL_BINS "$BIN_DIR/korean_val.bin"
fi

echo ""
echo "=== 완료 ==="
echo "학습 데이터: $BIN_DIR/korean_train.bin"
echo "검증 데이터: $BIN_DIR/korean_val.bin"
echo "토크나이저:  $TOKENIZER_DIR/tokenizer.json"
echo ""
echo "다음 단계:"
echo "  python3 -c \""
echo "  import numpy as np"
echo "  d = np.memmap('$BIN_DIR/korean_train.bin', dtype='uint16', mode='r')"
echo "  print(f'총 토큰: {len(d):,}')"
echo "  \""
