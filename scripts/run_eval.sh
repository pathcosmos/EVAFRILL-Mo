#!/usr/bin/env bash
# Usage: bash scripts/run_eval.sh <checkpoint_dir>
# Example: bash scripts/run_eval.sh checkpoints/korean_1b_fp8_run1/checkpoint-0200000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT="${1:?Usage: bash scripts/run_eval.sh <checkpoint_dir>}"

echo "=== Perplexity Evaluation ==="
python "$PROJECT_DIR/eval/perplexity.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$PROJECT_DIR/data/korean_val.bin" \
  --device cuda:0

echo ""
echo "=== Text Generation ==="
python "$PROJECT_DIR/eval/generate.py" \
  --checkpoint "$CHECKPOINT" \
  --prompt "안녕하세요, 저는" \
  --max_new_tokens 200 \
  --device cuda:0
