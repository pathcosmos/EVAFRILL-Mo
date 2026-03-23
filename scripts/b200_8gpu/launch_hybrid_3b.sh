#!/bin/bash
# ============================================================================
# FRANKENSTALLM-H 3B: Hybrid Mamba-2 + Transformer 학습 런치 스크립트
# ============================================================================
#
# 사용법:
#   nohup setsid bash scripts/launch_hybrid_3b.sh > logs/hybrid_3b.log 2>&1 &
#
# SIGHUP 방어: nohup + setsid 조합으로 SSH 끊김에도 학습 유지
# ============================================================================

set -euo pipefail

# ---- 환경 변수 ----
export OMP_NUM_THREADS=4
export NCCL_ALGO=NVLS           # NVSwitch 최적 알고리즘
export NCCL_IB_DISABLE=1        # InfiniBand 비활성 (단일 노드)
export NCCL_P2P_LEVEL=NVL       # NVLink P2P
export NCCL_NET_GDR_LEVEL=0     # GPU Direct RDMA 비활성 (단일 노드)

# ---- 경로 ----
PROJECT_ROOT="/PROJECT/0325120031_A/ghong/taketimes/llm-bang"
CONFIG="${PROJECT_ROOT}/configs/hybrid_3b.yaml"
TRAIN_DATA="${PROJECT_ROOT}/data/3b_train.bin"
VAL_DATA="${PROJECT_ROOT}/data/3b_val.bin"
CKPT_DIR="${PROJECT_ROOT}/checkpoints/hybrid_3b_run1"
LOG_FILE="${PROJECT_ROOT}/logs/hybrid_3b_train.log"

# ---- 디렉토리 생성 ----
mkdir -p "${CKPT_DIR}"
mkdir -p "$(dirname ${LOG_FILE})"

cd "${PROJECT_ROOT}"

echo "============================================"
echo "  FRANKENSTALLM-H 3B Hybrid Training"
echo "  Config: ${CONFIG}"
echo "  Data: ${TRAIN_DATA}"
echo "  Checkpoint: ${CKPT_DIR}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ---- 학습 실행 (8 GPU DDP) ----
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train/pretrain.py \
    --config "${CONFIG}" \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --batch_size 4 \
    --lr 2e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --grad_accum 8 \
    --max_steps 57000 \
    --log_file "${LOG_FILE}" \
    --use_fp8 \
    "$@"

echo "Training finished at $(date '+%Y-%m-%d %H:%M:%S')"
