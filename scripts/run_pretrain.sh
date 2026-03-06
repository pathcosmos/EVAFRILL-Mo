#!/bin/bash
# Usage: bash scripts/run_pretrain.sh [additional torchrun args]
# Runs 8-GPU DDP pretraining via torchrun.
#
# Any extra arguments are forwarded verbatim to pretrain.py.
# Examples:
#   bash scripts/run_pretrain.sh --max_steps 200000
#   bash scripts/run_pretrain.sh --resume checkpoints/checkpoint-0010000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  "$PROJECT_DIR/train/pretrain.py" \
  --config "$PROJECT_DIR/configs/small.yaml" \
  --train_data "$PROJECT_DIR/data/train.bin" \
  --val_data "$PROJECT_DIR/data/val.bin" \
  --checkpoint_dir "$PROJECT_DIR/checkpoints" \
  --batch_size 8 \
  --grad_accum 4 \
  --warmup_steps 2000 \
  "$@"
