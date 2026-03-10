#!/bin/bash
# ==========================================================
# EVAFRILL-Mo 3B SFT — Resilient Training Wrapper
# - Base: checkpoints/3b_final/checkpoint-0319772
# - Data: data/sft_combined/train_filtered.jsonl (2.44M)
# - ~1 epoch (44,000 steps), 7× B200, eff_batch=56
# - Automatically restarts from latest checkpoint on crash
# ==========================================================

set -u

# ---- ai-env 자동 활성화 ------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AIENV_DIR="${SCRIPT_DIR}/ai-env"
if [ -d "$AIENV_DIR" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    source "$AIENV_DIR/bin/activate"
    echo "[INFO] ai-env activated: $(which python)"
fi

BASE_CHECKPOINT="checkpoints/3b_final/checkpoint-0319772"
SFT_DATA="data/sft_combined/train_filtered.jsonl"
VAL_DATA="data/sft_combined/val_filtered.jsonl"
CKPT_DIR="checkpoints/3b_sft"
LOG_FILE="$CKPT_DIR/train.log"
NOHUP_OUT="$CKPT_DIR/nohup.out"
MONITOR_LOG="$CKPT_DIR/monitor.log"
CONFIG="configs/korean_3b_sft.yaml"
MAX_RETRIES=10
RETRY_DELAY=30
PORT=29530

# ---- 환경 설정 (이전 110-step 정상 실행 확인된 최소 설정) --------------------
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$CKPT_DIR"

log_event() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

find_latest_sft_checkpoint() {
    local latest=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    echo "$latest"
}

cleanup_gpu() {
    log_event "Cleaning up GPU processes..."
    pkill -9 -f "sft.py" 2>/dev/null
    sleep 5
    local gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$gpu_used" -gt 500 ]; then
        log_event "WARNING: GPUs still in use (${gpu_used} MiB). Waiting 15s..."
        sleep 15
    fi
}

run_training() {
    local attempt=$1
    local resume_arg=""

    local latest_ckpt=$(find_latest_sft_checkpoint)
    if [ -n "$latest_ckpt" ]; then
        resume_arg="--resume $latest_ckpt"
        log_event "SFT 재시작: $latest_ckpt"
    else
        log_event "SFT 최초 시작 (base: $BASE_CHECKPOINT)"
    fi

    log_event "=== SFT attempt $attempt/$MAX_RETRIES ==="

    torchrun --nproc_per_node=7 --master_port=$PORT \
        train/sft.py \
        --base_checkpoint "$BASE_CHECKPOINT" \
        --sft_data "$SFT_DATA" \
        --val_data "$VAL_DATA" \
        --checkpoint_dir "$CKPT_DIR" \
        --config "$CONFIG" \
        --log_file "$LOG_FILE" \
        --use_fp8 \
        --seed 42 \
        $resume_arg \
        >> "$NOHUP_OUT" 2>&1

    return $?
}

# ============================================================
# Main loop
# ============================================================
log_event "=========================================="
log_event "EVAFRILL-Mo 3B SFT resilient wrapper 시작"
log_event "Base: $BASE_CHECKPOINT"
log_event "Data: $SFT_DATA (2.44M samples)"
log_event "Target: 44,000 steps (~1 epoch, eff_batch=56)"
log_event "GPUs: 7× B200"
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

    PORT=$((PORT + 1))
    log_event "다음 시도: port $PORT"
    log_event "${RETRY_DELAY}s 대기 후 재시작..."
    sleep "$RETRY_DELAY"
done
