#!/usr/bin/env bash
# =============================================================================
# launch_korean_1b.sh — 8-GPU FP8 pretraining launcher for 1B Korean LLM
#
# Usage:
#   bash scripts/launch_korean_1b.sh                     # full run
#   bash scripts/launch_korean_1b.sh --max_steps 500     # quick test
#   bash scripts/launch_korean_1b.sh --resume checkpoints/korean_1b_fp8_run1/checkpoint-0010000
#
# Config is read from configs/korean_1b_fp8.yaml (model) + CLI args (train).
# Effective batch size: 8 (local) × 8 GPU × 4 (grad_accum) × 4096 (seq_len)
#                     = 1,048,576 tokens / step
# Logs: checkpoints/<RUN_NAME>/train.log
#       checkpoints/<RUN_NAME>/tensorboard/
# =============================================================================
set -euo pipefail

# ---- Configurable defaults --------------------------------------------------
RUN_NAME="${RUN_NAME:-korean_1b_fp8_run1}"
CONFIG="${CONFIG:-configs/korean_1b_fp8.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/korean_train.bin}"
VAL_DATA="${VAL_DATA:-data/korean_val.bin}"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="${CKPT_DIR}/train.log"
NPROC=8
MASTER_PORT="${MASTER_PORT:-29501}"

# ---- Defaults that can be overridden via extra CLI args --------------------
MAX_STEPS=34000     # 4 에포크 × 8.91B tokens = 35.6B (Muennighoff 2023: 4에포크 초과 시 val loss 상승)
BATCH_SIZE=8
GRAD_ACCUM=4
WARMUP_STEPS=2000   # 34k steps의 5.9% (기존 4000 = 11.8%로 과도)
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
cd "$(dirname "$0")/.."  # always run from project root

# ---- Pre-flight check: Korean data must exist before launching --------------
if [[ ! -f "${TRAIN_DATA}" ]]; then
    echo "=================================================================="
    echo "  ERROR: Training data not found: ${TRAIN_DATA}"
    echo ""
    echo "  You need to run the Korean data pipeline first."
    echo "  Example steps:"
    echo "    1. Download / prepare raw Korean corpus"
    echo "    2. Tokenise and pack into binary format:"
    echo "         python data/prepare_korean.py --output data/korean_train.bin"
    echo "    3. Re-run this script once the file exists."
    echo "=================================================================="
    exit 1
fi

if [[ ! -f "${VAL_DATA}" ]]; then
    echo "=================================================================="
    echo "  ERROR: Validation data not found: ${VAL_DATA}"
    echo ""
    echo "  You need to run the Korean data pipeline first."
    echo "  Example steps:"
    echo "    1. Download / prepare raw Korean corpus"
    echo "    2. Tokenise and pack into binary format (val split):"
    echo "         python data/prepare_korean.py --output_val data/korean_val.bin"
    echo "    3. Re-run this script once the file exists."
    echo "=================================================================="
    exit 1
fi

mkdir -p "${CKPT_DIR}"

echo "=================================================================="
echo "  Run name    : ${RUN_NAME}"
echo "  Config      : ${CONFIG}"
echo "  Train data  : ${TRAIN_DATA}"
echo "  Val data    : ${VAL_DATA}"
echo "  CKPT dir    : ${CKPT_DIR}"
echo "  Log file    : ${LOG_FILE}"
echo "  Max steps   : ${MAX_STEPS}"
echo "  Batch size  : ${BATCH_SIZE} (local) × ${NPROC} GPU × ${GRAD_ACCUM} grad_accum"
echo "  Warmup      : ${WARMUP_STEPS} steps"
echo "  Master port : ${MASTER_PORT}"
echo "  Started     : $(date)"
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
         | grep -v "self.m.impl" \
         | tee -a "${LOG_FILE}"

echo "=================================================================="
echo "  Done : $(date)"
echo "=================================================================="
