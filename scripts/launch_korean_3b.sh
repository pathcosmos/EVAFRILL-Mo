#!/usr/bin/env bash
# =============================================================================
# launch_korean_3b.sh — 8-GPU FP8 pretraining launcher for 3B Korean LLM
#
# Usage:
#   bash scripts/launch_korean_3b.sh                     # full run (~60B tokens)
#   bash scripts/launch_korean_3b.sh --max_steps 50      # quick benchmark
#   bash scripts/launch_korean_3b.sh --resume checkpoints/korean_3b_fp8_run1/checkpoint-XXXXX
#
# Effective batch size: 8 (local) × 8 GPU × 4 (grad_accum) × 4096 (seq_len)
#                     = 1,048,576 tokens / step
# =============================================================================
set -euo pipefail

RUN_NAME="${RUN_NAME:-korean_3b_fp8_run1}"
CONFIG="${CONFIG:-configs/3b_pretrain.yaml}"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="${CKPT_DIR}/train.log"
NPROC=8
MASTER_PORT="${MASTER_PORT:-29502}"

MAX_STEPS=57000
BATCH_SIZE=4
GRAD_ACCUM=8
LR=1.5e-4
WARMUP_STEPS=2000
SEED=42

EXTRA_ARGS="$@"

# ---- B200 / NVSwitch NCCL tuning -------------------------------------------
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_BUFFSIZE=67108864
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# cd FIRST — 이후 상대경로 체크가 프로젝트 루트 기준으로 동작
cd "$(dirname "$0")/.."

# TRAIN_DATA fallback: cd 이후에 상대경로 체크
if [[ -f "data/merged_3b_train.bin" ]]; then
    TRAIN_DATA="${TRAIN_DATA:-data/merged_3b_train.bin}"
    echo "Using merged training data: data/merged_3b_train.bin"
elif [[ -f "data/korean_train.bin" ]]; then
    TRAIN_DATA="${TRAIN_DATA:-data/korean_train.bin}"
    echo "Using fallback training data: data/korean_train.bin"
else
    echo "ERROR: No training data found (data/merged_3b_train.bin or data/korean_train.bin)"
    exit 1
fi

# VAL_DATA fallback: cd 이후에 상대경로 체크
VAL_DATA="${VAL_DATA:-data/merged_3b_val.bin}"
if [[ ! -f "${VAL_DATA}" ]]; then
    VAL_DATA="data/korean_val.bin"
fi

if [[ ! -f "${TRAIN_DATA}" ]]; then
    echo "ERROR: Training data not found: ${TRAIN_DATA}"
    exit 1
fi
if [[ ! -f "${VAL_DATA}" ]]; then
    echo "ERROR: Validation data not found: ${VAL_DATA}"
    exit 1
fi

mkdir -p "${CKPT_DIR}"

echo "=================================================================="
echo "  Run name    : ${RUN_NAME}"
echo "  Config      : ${CONFIG}"
echo "  Train data  : ${TRAIN_DATA}"
echo "  CKPT dir    : ${CKPT_DIR}"
echo "  Max steps   : ${MAX_STEPS}"
echo "  LR          : ${LR}"
echo "  Batch size  : ${BATCH_SIZE} (local) × ${NPROC} GPU × ${GRAD_ACCUM} grad_accum"
echo "  Started     : $(date)"
echo "=================================================================="

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
    --lr ${LR} \
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
