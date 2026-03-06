#!/usr/bin/env bash
# =============================================================================
# launch_3b_sft.sh — 8-GPU FP8 SFT launcher for 3B Korean LLM
#
# Usage:
#   bash scripts/launch_3b_sft.sh
#   bash scripts/launch_3b_sft.sh --max_steps 200    # quick test
#   bash scripts/launch_3b_sft.sh --resume checkpoints/korean_3b_sft_v1/checkpoint-0002000
#
# Base model : checkpoints/korean_3b_fp8_run1/checkpoint-XXXXXX  (기본값)
#              --base_checkpoint 인자로 덮어쓸 수 있음
# SFT data   : data/sft_combined/train_filtered.jsonl
#              (먼저 scripts/prepare_sft_combined.sh → data/filter_sft_v2.py 실행)
#
# Effective batch: 2 (local) × 8 GPU × 4 (grad_accum) = 64 samples/step
# =============================================================================
set -euo pipefail

# ---- Configurable defaults --------------------------------------------------
RUN_NAME="${RUN_NAME:-korean_3b_sft_v1}"
CONFIG="${CONFIG:-configs/korean_3b_sft.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoints/korean_3b_fp8_run1/checkpoint-0057000}"
SFT_DATA="${SFT_DATA:-data/sft_combined/train_filtered.jsonl}"
VAL_DATA="${VAL_DATA:-data/sft_combined/val_filtered.jsonl}"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="${CKPT_DIR}/train.log"
NPROC=8
MASTER_PORT="${MASTER_PORT:-29503}"

MAX_STEPS=33000
BATCH_SIZE=2
GRAD_ACCUM=4
LR="1.0e-5"
WARMUP_STEPS=500
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

# 3B 모델 VRAM 절약 — 동적 메모리 세그먼트 확장 허용
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

# ---- Pre-flight checks ------------------------------------------------------
if [[ ! -d "${BASE_CHECKPOINT}" ]]; then
    echo "=================================================================="
    echo "  ERROR: Base checkpoint 디렉토리를 찾을 수 없습니다."
    echo "  경로: ${BASE_CHECKPOINT}"
    echo ""
    echo "  --base_checkpoint 인자로 실제 경로를 지정하거나"
    echo "  BASE_CHECKPOINT 환경변수를 설정하세요."
    echo "  예: bash scripts/launch_3b_sft.sh --base_checkpoint checkpoints/korean_3b_fp8_run1/checkpoint-0057000"
    echo "=================================================================="
    exit 1
fi

if [[ ! -f "${SFT_DATA}" ]]; then
    echo "=================================================================="
    echo "  ERROR: SFT 학습 데이터를 찾을 수 없습니다: ${SFT_DATA}"
    echo ""
    echo "  데이터 준비 순서:"
    echo "    1. bash scripts/prepare_sft_combined.sh"
    echo "    2. python data/filter_sft_v2.py \\"
    echo "           --input  data/sft_combined/train.jsonl \\"
    echo "           --output data/sft_combined/train_filtered.jsonl"
    echo "=================================================================="
    exit 1
fi

# val 파일 없으면 원본 val.jsonl 로 폴백
if [[ ! -f "${VAL_DATA}" ]]; then
    VAL_FALLBACK="data/sft_combined/val.jsonl"
    if [[ -f "${VAL_FALLBACK}" ]]; then
        VAL_DATA="${VAL_FALLBACK}"
        echo "[INFO] val_filtered 없음, 폴백: ${VAL_DATA}"
    else
        echo "ERROR: VAL_DATA 파일을 찾을 수 없습니다: ${VAL_DATA}"
        exit 1
    fi
fi

mkdir -p "${CKPT_DIR}"

echo "=================================================================="
echo "  3B SFT Fine-Tuning"
echo "  Run name        : ${RUN_NAME}"
echo "  Config          : ${CONFIG}"
echo "  Base checkpoint : ${BASE_CHECKPOINT}"
echo "  SFT data        : ${SFT_DATA}"
echo "  Val data        : ${VAL_DATA}"
echo "  CKPT dir        : ${CKPT_DIR}"
echo "  Log file        : ${LOG_FILE}"
echo "  Max steps       : ${MAX_STEPS}"
echo "  Batch size      : ${BATCH_SIZE} (local) × ${NPROC} GPU × ${GRAD_ACCUM} grad_accum = $((BATCH_SIZE * NPROC * GRAD_ACCUM)) eff_batch"
echo "  Learning rate   : ${LR}"
echo "  Warmup          : ${WARMUP_STEPS} steps"
echo "  Master port     : ${MASTER_PORT}"
echo "  ALLOC_CONF      : ${PYTORCH_CUDA_ALLOC_CONF}"
echo "  Started         : $(date)"
echo "=================================================================="

export PYTHONWARNINGS="ignore::UserWarning:torch.library"

torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    train/sft.py \
    --config "${CONFIG}" \
    --base_checkpoint "${BASE_CHECKPOINT}" \
    --sft_data "${SFT_DATA}" \
    --val_data "${VAL_DATA}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --log_file "${LOG_FILE}" \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --lr ${LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    --use_fp8 \
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
echo "  3B SFT Done : $(date)"
echo "=================================================================="
