#!/usr/bin/env bash
# =============================================================================
# launch_fp8.sh — 8-GPU FP8 pretraining launcher for B200
#
# Usage:
#   bash scripts/launch_fp8.sh                    # full run
#   bash scripts/launch_fp8.sh --max_steps 500    # quick test
#   bash scripts/launch_fp8.sh --resume checkpoints/small_fp8_run1/checkpoint-0001000
#
# Config is read from configs/small_fp8.yaml (model) + CLI args (train).
# Logs: checkpoints/<RUN_NAME>/train.log
#       checkpoints/<RUN_NAME>/tensorboard/
# =============================================================================
set -euo pipefail

# ---- Configurable defaults --------------------------------------------------
RUN_NAME="${RUN_NAME:-small_fp8_run1}"
CONFIG="${CONFIG:-configs/small_fp8.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/train.bin}"
VAL_DATA="${VAL_DATA:-data/val.bin}"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="${CKPT_DIR}/train.log"
NPROC=8
MASTER_PORT="${MASTER_PORT:-29500}"

# ---- Defaults that can be overridden via extra CLI args --------------------
MAX_STEPS=100000
BATCH_SIZE=8
GRAD_ACCUM=4
WARMUP_STEPS=2000
SEED=42

# ---- Pass remaining CLI args directly to pretrain.py ----------------------
EXTRA_ARGS="$@"

# ---- B200 / NVSwitch single-node NCCL tuning --------------------------------
# Single-node NVSwitch (NV18 full-mesh): disable IB to prevent NCCL probing.
export NCCL_IB_DISABLE=1
# Use Ring algorithm for large gradient tensors (128M-70B model range).
export NCCL_ALGO=Ring
# Simple protocol is optimal for NVLink bulk transfers (vs LL/LL128 for IB).
export NCCL_PROTO=Simple
# More channels → better NVSwitch saturation for large all-reduce payloads.
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
# Larger NCCL buffer (64 MB) reduces ring synchronisation overhead.
export NCCL_BUFFSIZE=67108864
# CPU thread limits (72 cores ÷ 8 ranks = 9; use 4 for DataLoader headroom).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ---- Setup ------------------------------------------------------------------
mkdir -p "${CKPT_DIR}"
cd "$(dirname "$0")/.."  # always run from project root

echo "=================================================================="
echo "  Run name  : ${RUN_NAME}"
echo "  Config    : ${CONFIG}"
echo "  CKPT dir  : ${CKPT_DIR}"
echo "  Log file  : ${LOG_FILE}"
echo "  Started   : $(date)"
echo "=================================================================="

# Suppress the harmless flash_attn kernel override warning from all ranks.
export PYTHONWARNINGS="ignore::UserWarning:torch.library"

torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    train/pretrain.py \
    --config "${CONFIG}" \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --log_file "${LOG_FILE}" \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    ${EXTRA_ARGS} \
    2>&1 | grep -v "UserWarning" \
         | grep -v "Warning only once" \
         | grep -v "Overriding a previously" \
         | grep -v "dispatch key:" \
         | grep -v "previous kernel:" \
         | grep -v "new kernel:" \
         | grep -v "operator: flash_attn" \
         | grep -v "registered at /usr/local" \
         | grep -v "self.m.impl"

echo "=================================================================="
echo "  Done : $(date)"
echo "=================================================================="
