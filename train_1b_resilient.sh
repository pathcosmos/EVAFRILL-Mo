#!/bin/bash
# ==========================================================
# Resilient 1B Training Wrapper
# - Automatically restarts from latest checkpoint on crash
# - Logs all events to a separate monitor log
# - Max retry limit to avoid infinite loops
# ==========================================================

set -u

CONFIG="/tmp/bench_1b.yaml"
TRAIN_DATA="data/3b_train.bin"
CKPT_DIR="checkpoints/1b_final"
LOG_FILE="$CKPT_DIR/train.log"
NOHUP_OUT="$CKPT_DIR/nohup.out"
MONITOR_LOG="$CKPT_DIR/monitor.log"
MAX_RETRIES=10
RETRY_DELAY=30  # seconds between retries
PORT=29502

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$CKPT_DIR"

log_event() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

find_latest_checkpoint() {
    # Find the most recent checkpoint-XXXXXXX directory
    local latest=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    echo "$latest"
}

cleanup_gpu() {
    log_event "Cleaning up GPU processes..."
    pkill -9 -f "pretrain.py.*$CKPT_DIR" 2>/dev/null
    sleep 5
    # Verify GPUs are free
    local gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$gpu_used" -gt 500 ]; then
        log_event "WARNING: GPUs still in use (${gpu_used} MiB). Waiting 15s..."
        sleep 15
    fi
}

run_training() {
    local attempt=$1
    local resume_arg=""

    # Check for existing checkpoint to resume from
    local latest_ckpt=$(find_latest_checkpoint)
    if [ -n "$latest_ckpt" ]; then
        resume_arg="--resume $latest_ckpt"
        log_event "Resuming from checkpoint: $latest_ckpt"
    else
        log_event "Starting fresh training (no checkpoint found)"
    fi

    log_event "=== Training attempt $attempt/$MAX_RETRIES ==="
    log_event "Command: torchrun --nproc_per_node=7 --master_port=$PORT train/pretrain.py --config $CONFIG $resume_arg"

    torchrun --nproc_per_node=7 --master_port=$PORT \
        train/pretrain.py \
        --config "$CONFIG" \
        --train_data "$TRAIN_DATA" \
        --batch_size 16 \
        --lr 3e-4 \
        --weight_decay 0.1 \
        --warmup_steps 915 \
        --grad_accum 1 \
        --max_steps 45776 \
        --checkpoint_dir "$CKPT_DIR" \
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
log_event "Resilient training wrapper started"
log_event "Max retries: $MAX_RETRIES"
log_event "=========================================="

for attempt in $(seq 1 $MAX_RETRIES); do
    run_training "$attempt"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_event "Training completed successfully!"
        exit 0
    fi

    log_event "Training crashed with exit code $exit_code"

    # Check if this was the last attempt
    if [ $attempt -eq $MAX_RETRIES ]; then
        log_event "FATAL: Max retries ($MAX_RETRIES) exhausted. Giving up."
        exit 1
    fi

    # Cleanup before retry
    cleanup_gpu

    # Increment port to avoid EADDRINUSE
    PORT=$((PORT + 1))
    log_event "Next attempt will use port $PORT"

    log_event "Waiting ${RETRY_DELAY}s before retry..."
    sleep "$RETRY_DELAY"
done
