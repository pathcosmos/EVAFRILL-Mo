#!/usr/bin/env bash
# =============================================================================
# training_watchdog.sh — FRANKENSTALLM 3B Cron-based Training Watchdog
# Run: every 10 minutes via cron
# Alerts via Telegram only when problems are detected.
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CKPT_DIR="$WORKDIR/checkpoints/korean_3b_fp8_run1"
LOG_FILE="$CKPT_DIR/train.log"
PID_FILE="$CKPT_DIR/train.pid"
WATCHDOG_LOG="$CKPT_DIR/watchdog.log"
STATE_FILE="$CKPT_DIR/watchdog.state"   # persists last-good step/time
NOTIFY="python3 $WORKDIR/scripts/telegram_notify.py"

# ─── Thresholds ──────────────────────────────────────────────────────────────
LOSS_SPIKE_THRESHOLD="5.0"       # alert if loss > this value
LOSS_NAN_PATTERN="nan|inf|NaN|Inf"
STALL_SECONDS=900                # 15 min without new log line → stalled
DISK_WARN_PCT=85                 # alert if disk usage >= this %
GPU_UTIL_WARN_PCT=20             # alert if avg GPU util drops below this %
MIN_TOKPS=5000                   # alert if tok/s drops below this
TOTAL_STEPS=57000
WAIT_COUNT_FILE="/tmp/frankenstallm-wait-count"  # 대기 횟수 파일
MAX_WAIT_COUNT=10                                  # 이 횟수 초과 시 알림 후 cron 해제

# ─── Helpers ─────────────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }

log_msg() {
    echo "[$(ts)] $*"
}

send_alert() {
    local level="$1"
    local msg="$2"
    log_msg "ALERT[$level]: $msg"
    $NOTIFY "<b>[FRANKENSTALLM ALERT] $level</b>

$msg

<i>$(ts) | watchdog check</i>" || true
}

# ─── 1. Process alive check ──────────────────────────────────────────────────
check_process() {
    if [[ ! -f "$PID_FILE" ]]; then
        # 대기 모드: PID 파일 없으면 학습 미시작 상태로 카운트
        local wait_count=0
        [[ -f "$WAIT_COUNT_FILE" ]] && wait_count=$(cat "$WAIT_COUNT_FILE" 2>/dev/null || echo 0)
        wait_count=$(( wait_count + 1 ))
        echo "$wait_count" > "$WAIT_COUNT_FILE"
        log_msg "Training not started yet (waiting ${wait_count}/${MAX_WAIT_COUNT})."

        if (( wait_count > MAX_WAIT_COUNT )); then
            send_alert "WAIT_TIMEOUT" "학습이 <b>${wait_count}회</b> 체크 동안 시작되지 않았습니다 (~$((wait_count * 10))분).

PID 파일 없음: <code>$PID_FILE</code>

Watchdog cron을 자동 해제합니다. 학습 시작 후 직접 재등록하세요:
<code>crontab -e</code>"
            # cron에서 training_watchdog 제거
            crontab -l 2>/dev/null | grep -v "training_watchdog" | crontab -
            rm -f "$WAIT_COUNT_FILE"
            log_msg "Watchdog cron entry removed after ${wait_count} waits."
        fi
        return 1
    fi
    # 학습 시작됨 → 대기 카운터 초기화
    rm -f "$WAIT_COUNT_FILE"

    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')

    if [[ -z "$pid" ]]; then
        send_alert "PROCESS" "PID file is empty: $PID_FILE"
        return 1
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
        # Check if it completed normally (step == TOTAL_STEPS)
        local last_step
        last_step=$(grep -oP 'step\s+\K[0-9]+' "$LOG_FILE" 2>/dev/null | tail -1)
        if [[ "$last_step" == "$TOTAL_STEPS" ]]; then
            log_msg "Training COMPLETED at step $TOTAL_STEPS — process exit is expected."
            send_alert "COMPLETE" "Training completed normally at step <code>$TOTAL_STEPS/$TOTAL_STEPS</code>."
        else
            send_alert "CRASH" "Training process (PID $pid) is NOT running.
Last logged step: <code>${last_step:-unknown}</code>/$TOTAL_STEPS

Check log: <code>tail -50 $LOG_FILE</code>"
        fi
        return 1
    fi

    log_msg "Process PID $pid is alive."
    return 0
}

# ─── 2. Stall detection ──────────────────────────────────────────────────────
check_stall() {
    if [[ ! -f "$LOG_FILE" ]]; then
        send_alert "STALL" "Log file not found: $LOG_FILE"
        return 1
    fi

    local log_mtime now elapsed
    log_mtime=$(stat -c '%Y' "$LOG_FILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    elapsed=$(( now - log_mtime ))

    if (( elapsed >= STALL_SECONDS )); then
        local mins=$(( elapsed / 60 ))
        send_alert "STALL" "No log activity for <b>${mins} minutes</b> (threshold: $(( STALL_SECONDS/60 ))min).
Log last modified: <code>$(date -d "@$log_mtime" '+%Y-%m-%d %H:%M:%S')</code>
Training may be hung or extremely slow."
        return 1
    fi

    log_msg "Log freshness OK: last update ${elapsed}s ago."
    return 0
}

# ─── 3. Loss anomaly check ───────────────────────────────────────────────────
check_loss() {
    if [[ ! -f "$LOG_FILE" ]]; then
        return 0
    fi

    # Get last step line
    local last_line
    last_line=$(grep -E 'step\s+[0-9]+.*loss' "$LOG_FILE" 2>/dev/null | tail -1)

    if [[ -z "$last_line" ]]; then
        log_msg "No step lines found in log yet."
        return 0
    fi

    local loss step
    loss=$(echo "$last_line" | grep -oP 'loss\s+\K[0-9.eE+\-naifNIF]+' || echo "")
    step=$(echo "$last_line" | grep -oP 'step\s+\K[0-9]+' || echo "0")

    if [[ -z "$loss" ]]; then
        log_msg "Could not parse loss from: $last_line"
        return 0
    fi

    # NaN/Inf check
    if echo "$loss" | grep -qiE "$LOSS_NAN_PATTERN"; then
        send_alert "LOSS_NAN" "Loss is <b>$loss</b> at step <code>$step</code>.
Training has diverged — NaN/Inf detected.

Last log line:
<code>${last_line}</code>"
        return 1
    fi

    # Spike check (only after warmup, step > 500)
    if (( step > 500 )); then
        local loss_int
        loss_int=$(echo "$loss >= $LOSS_SPIKE_THRESHOLD" | bc -l 2>/dev/null || echo 0)
        if [[ "$loss_int" == "1" ]]; then
            send_alert "LOSS_SPIKE" "Loss spike detected: <b>$loss</b> at step <code>$step</code> (threshold: $LOSS_SPIKE_THRESHOLD).

Last log line:
<code>${last_line}</code>"
            return 1
        fi
    fi

    log_msg "Loss OK: $loss at step $step."
    return 0
}

# ─── 4. Throughput check ─────────────────────────────────────────────────────
check_throughput() {
    if [[ ! -f "$LOG_FILE" ]]; then
        return 0
    fi

    local last_line
    last_line=$(grep -E 'step\s+[0-9]+.*tok/s' "$LOG_FILE" 2>/dev/null | tail -1)
    [[ -z "$last_line" ]] && return 0

    # tok/s may be formatted with commas: 36,321
    local tokps step
    tokps=$(echo "$last_line" | grep -oP 'tok/s\s+\K[\d,]+' | tr -d ',' || echo "")
    step=$(echo "$last_line" | grep -oP 'step\s+\K[0-9]+' || echo "0")

    if [[ -z "$tokps" ]]; then
        log_msg "Could not parse tok/s from last log line."
        return 0
    fi

    if (( step > 100 && tokps < MIN_TOKPS )); then
        send_alert "THROUGHPUT" "Throughput dropped to <b>${tokps} tok/s</b> at step <code>$step</code> (min: ${MIN_TOKPS}).
GPU may be throttling, NCCL stalled, or a data worker is slow."
        return 1
    fi

    log_msg "Throughput OK: ${tokps} tok/s at step $step."
    return 0
}

# ─── 5. GPU utilization check ────────────────────────────────────────────────
check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        log_msg "nvidia-smi not available — skipping GPU check."
        return 0
    fi

    local avg_util
    avg_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print 0}')

    if [[ -z "$avg_util" || "$avg_util" == "0" ]]; then
        log_msg "GPU util query returned 0 or empty — possibly all idle."
        # Only alert if process is also running
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            send_alert "GPU_IDLE" "All 8× B200 GPUs show <b>0% utilization</b> while training process is alive.
Possible NCCL hang or data pipeline stall."
            return 1
        fi
        return 0
    fi

    if (( avg_util < GPU_UTIL_WARN_PCT )); then
        local gpu_details
        gpu_details=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader 2>/dev/null | head -8 || echo "unavailable")
        send_alert "GPU_LOW" "Average GPU utilization: <b>${avg_util}%</b> (threshold: ${GPU_UTIL_WARN_PCT}%).

GPU details:
<code>${gpu_details}</code>"
        return 1
    fi

    log_msg "GPU utilization OK: ${avg_util}% average."
    return 0
}

# ─── 6. Disk space check ─────────────────────────────────────────────────────
check_disk() {
    local usage_pct
    usage_pct=$(df "$CKPT_DIR" 2>/dev/null | awk 'NR==2 {gsub(/%/,"",$5); print $5}')

    if [[ -z "$usage_pct" ]]; then
        log_msg "Could not determine disk usage for $CKPT_DIR."
        return 0
    fi

    if (( usage_pct >= DISK_WARN_PCT )); then
        local avail
        avail=$(df -h "$CKPT_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
        send_alert "DISK" "Disk usage at <b>${usage_pct}%</b> (threshold: ${DISK_WARN_PCT}%).
Available: <b>${avail}</b> on partition containing checkpoints.

Risk: checkpoint saves may fail. Consider deleting old checkpoints."
        return 1
    fi

    log_msg "Disk usage OK: ${usage_pct}% used."
    return 0
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    log_msg "=== Watchdog check START ==="

    local issues=0

    check_process  || (( issues++ )) || true
    check_stall    || (( issues++ )) || true
    check_loss     || (( issues++ )) || true
    check_throughput || (( issues++ )) || true
    check_gpu      || (( issues++ )) || true
    check_disk     || (( issues++ )) || true

    if (( issues == 0 )); then
        log_msg "All checks passed — no alerts sent."
    else
        log_msg "Watchdog found $issues issue(s) — alerts sent."
    fi

    log_msg "=== Watchdog check END ==="
}

main "$@"
