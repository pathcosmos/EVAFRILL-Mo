#!/bin/bash
# ==========================================================
# EVAFRILL-Mo 3B SFT — Single GPU (H100 MIG 3g.40gb)
# Fresh start from pretrained checkpoint
#
# - Base: checkpoints/3b_final/checkpoint-0319772
# - Data: data/sft_combined/train_filtered.jsonl (3.77M)
# - 1 epoch (135,000 steps), eff_batch=28, ~4,500 tok/s
# - BF16 + Gradient Checkpointing (no FP8, MIG 제약)
# - Automatically restarts from latest checkpoint on crash
# ==========================================================

set -u

# ---- ai-env 자동 활성화 ------------------------------------------------------
# /root/ai-env (Python 3.12) 사용: mamba_ssm 최적화 커널 포함
AIENV_DIR="/root/ai-env"
if [ -d "$AIENV_DIR" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$AIENV_DIR/bin/activate"
    echo "[INFO] ai-env activated: $(which python)"
fi

BASE_CHECKPOINT="checkpoints/3b_final/checkpoint-0319772"
SFT_DATA="data/sft_combined/train_filtered.jsonl"
VAL_DATA="data/sft_combined/val_filtered.jsonl"
CKPT_DIR="checkpoints/3b_sft_v2"
LOG_FILE="$CKPT_DIR/train.log"
NOHUP_OUT="$CKPT_DIR/nohup.out"
MONITOR_LOG="$CKPT_DIR/monitor.log"
CONFIG="configs/h100_mig/korean_3b_sft_1gpu.yaml"
MAX_RETRIES=10
RETRY_DELAY=30

# ---- 환경 설정 ---------------------------------------------------------------
export OMP_NUM_THREADS=4

mkdir -p "$CKPT_DIR"

log_event() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

find_latest_sft_checkpoint() {
    local latest=$(ls -d "$CKPT_DIR"/checkpoint-0* 2>/dev/null | sort -V | tail -1)
    echo "$latest"
}

cleanup_gpu() {
    log_event "Cleaning up GPU processes..."
    pkill -9 -f "sft.py" 2>/dev/null
    sleep 5
}

run_training() {
    local attempt=$1
    local resume_arg=""

    local latest_ckpt=$(find_latest_sft_checkpoint)
    if [ -n "$latest_ckpt" ]; then
        resume_arg="--resume $latest_ckpt"
        log_event "SFT 재시작: $latest_ckpt"
    else
        log_event "SFT 최초 시작 (fresh start from: $BASE_CHECKPOINT)"
    fi

    log_event "=== SFT attempt $attempt/$MAX_RETRIES ==="

    # Single GPU — BF16 + GradCkpt (no FP8, MIG 40GB)
    python train/sft.py \
        --base_checkpoint "$BASE_CHECKPOINT" \
        --sft_data "$SFT_DATA" \
        --val_data "$VAL_DATA" \
        --checkpoint_dir "$CKPT_DIR" \
        --config "$CONFIG" \
        --log_file "$LOG_FILE" \
        --device cuda:0 \
        --no_fp8 \
        --seed 42 \
        $resume_arg \
        >> "$NOHUP_OUT" 2>&1

    return $?
}

# ============================================================
# Main loop
# ============================================================
log_event "=========================================="
log_event "EVAFRILL-Mo 3B SFT (1-GPU) resilient wrapper 시작"
log_event "Base: $BASE_CHECKPOINT"
log_event "Data: $SFT_DATA (3.77M samples)"
log_event "Target: 135,000 steps (~1 epoch, eff_batch=28)"
log_event "GPU: H100 MIG 3g.40gb (42.3GB), BF16+GradCkpt"
log_event "Config: $CONFIG (bs=1, grad_accum=28, no_fp8)"
log_event "=========================================="

for attempt in $(seq 1 $MAX_RETRIES); do
    run_training "$attempt"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_event "SFT 학습 완료!"
        log_event "최종 체크포인트: $CKPT_DIR"
        exit 0
    fi

    log_event "학습 중단 (exit code: $exit_code)"

    if [ $attempt -eq $MAX_RETRIES ]; then
        log_event "FATAL: Max retries ($MAX_RETRIES) 소진. 종료."
        exit 1
    fi

    cleanup_gpu

    log_event "${RETRY_DELAY}s 대기 후 재시작..."
    sleep "$RETRY_DELAY"
done
