#!/usr/bin/env bash
# =============================================================================
# launch_sft.sh — 8-GPU FP8 SFT launcher for 1B Korean LLM
#
# Usage:
#   bash scripts/launch_sft.sh
#   bash scripts/launch_sft.sh --max_steps 500    # quick test
#   bash scripts/launch_sft.sh --resume checkpoints/korean_1b_sft/checkpoint-0001000
#
# Base model: checkpoints/korean_1b_fp8_run1/checkpoint-0034000
# SFT data:   data/sft/train.jsonl
# =============================================================================
set -euo pipefail

# ---- Configurable defaults --------------------------------------------------
RUN_NAME="${RUN_NAME:-korean_1b_sft}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoints/korean_1b_fp8_run1/checkpoint-0034000}"
SFT_DATA="${SFT_DATA:-data/sft/train.jsonl}"
VAL_DATA="${VAL_DATA:-data/sft/val.jsonl}"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="${CKPT_DIR}/train.log"
NPROC=8
MASTER_PORT="${MASTER_PORT:-29502}"

MAX_STEPS=9000
BATCH_SIZE=4
GRAD_ACCUM=2
LR="2.0e-5"
WARMUP_STEPS=300
SEED=42

EXTRA_ARGS="$@"

# ---- B200 / NVSwitch NCCL tuning (same as pretrain) -------------------------
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_BUFFSIZE=67108864
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "$(dirname "$0")/.."

# ---- Pre-flight checks ------------------------------------------------------
if [[ ! -d "${BASE_CHECKPOINT}" ]]; then
    echo "ERROR: Base checkpoint not found: ${BASE_CHECKPOINT}"
    exit 1
fi

if [[ ! -f "${SFT_DATA}" ]]; then
    echo "=================================================================="
    echo "  ERROR: SFT training data not found: ${SFT_DATA}"
    echo ""
    echo "  Run the data preparation script first:"
    echo "    python data/prepare_sft_data.py"
    echo "=================================================================="
    exit 1
fi

mkdir -p "${CKPT_DIR}"

echo "=================================================================="
echo "  SFT Fine-Tuning"
echo "  Run name       : ${RUN_NAME}"
echo "  Base checkpoint : ${BASE_CHECKPOINT}"
echo "  SFT data       : ${SFT_DATA}"
echo "  CKPT dir       : ${CKPT_DIR}"
echo "  Log file       : ${LOG_FILE}"
echo "  Max steps      : ${MAX_STEPS}"
echo "  Batch size     : ${BATCH_SIZE} (local) × ${NPROC} GPU × ${GRAD_ACCUM} grad_accum"
echo "  Learning rate  : ${LR}"
echo "  Warmup         : ${WARMUP_STEPS} steps"
echo "  Master port    : ${MASTER_PORT}"
echo "  Started        : $(date)"
echo "=================================================================="

export PYTHONWARNINGS="ignore::UserWarning:torch.library"

torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    train/sft.py \
    --base_checkpoint "${BASE_CHECKPOINT}" \
    --sft_data "${SFT_DATA}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --log_file "${LOG_FILE}" \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    --use_fp8 \
    --val_data "${VAL_DATA}" \
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
echo "  SFT Done : $(date)"
echo "=================================================================="
