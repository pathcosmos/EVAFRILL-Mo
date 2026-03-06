#!/bin/bash
# data/build_dataset.sh — Full pipeline: download → tokenizer → .bin
# Usage: bash data/build_dataset.sh [--langs "ko en"] [--ko_max 0] [--en_max 300000]
#
# Steps:
#   1. python data/download.py        → data/raw/*.txt
#   2. python tokenizer/train_tokenizer.py  → tokenizer/tokenizer.json
#   3. python data/prepare.py         → data/train.bin, data/val.bin

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default params
LANGS="ko en"
KO_MAX=0
EN_MAX=300000
VOCAB_SIZE=32000

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --langs) LANGS="$2"; shift 2 ;;
    --ko_max) KO_MAX="$2"; shift 2 ;;
    --en_max) EN_MAX="$2"; shift 2 ;;
    --vocab_size) VOCAB_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=============================="
echo " LLM-Bang Dataset Pipeline"
echo "=============================="
echo "  langs:      $LANGS"
echo "  ko_max:     $KO_MAX (0=all)"
echo "  en_max:     $EN_MAX"
echo "  vocab_size: $VOCAB_SIZE"
echo ""

# Step 1: Download
echo "[1/3] Downloading data..."
python data/download.py \
  --langs $LANGS \
  --ko_max $KO_MAX \
  --en_max $EN_MAX \
  --output_dir data/raw
echo ""

# Step 2: Train tokenizer
echo "[2/3] Training BPE tokenizer..."
python tokenizer/train_tokenizer.py \
  --input "data/raw/*.txt" \
  --output tokenizer/ \
  --vocab_size $VOCAB_SIZE
echo ""

# Step 3: Prepare .bin files
echo "[3/3] Tokenizing and saving .bin files..."
python data/prepare.py \
  --input "data/raw/*.txt" \
  --output data/train.bin \
  --val_output data/val.bin \
  --tokenizer tokenizer/tokenizer.json \
  --val_split 0.005
echo ""

echo "=============================="
echo " Done! Files:"
ls -lh data/*.bin 2>/dev/null || echo "  (no .bin files yet)"
echo "=============================="
