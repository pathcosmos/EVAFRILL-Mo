#!/usr/bin/env bash
# =============================================================================
# hourly_status.sh — FRANKENSTALLM 3B Hourly Training Status Report (Telegram)
# Run: every hour via cron
# Sends a rich formatted message with progress, loss, ETA, GPU/disk summary.
# =============================================================================
set -euo pipefail

# ─── Paths ───────────────────────────────────────────────────────────────────
WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CKPT_DIR="$WORKDIR/checkpoints/korean_3b_fp8_run1"
LOG_FILE="$CKPT_DIR/train.log"
PID_FILE="$CKPT_DIR/train.pid"
HOURLY_LOG="$CKPT_DIR/hourly_status.log"
NOTIFY="python3 $WORKDIR/scripts/telegram_notify.py"

TOTAL_STEPS=57000
TOTAL_TOKENS_B=114   # billion tokens target (57K steps × batch)

# ─── Helpers ─────────────────────────────────────────────────────────────────
ts()    { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(ts)] $*"; }

# Safely get last matching value from log
parse_last() {
    local pattern="$1"
    grep -oP "$pattern" "$LOG_FILE" 2>/dev/null | tail -1 || echo ""
}

# ─── Parse training log ───────────────────────────────────────────────────────
parse_log() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "NO_LOG"
        return 1
    fi

    # Get the last step line
    LAST_LINE=$(grep -E 'step\s+[0-9]+.*loss' "$LOG_FILE" 2>/dev/null | tail -1 || echo "")
    if [[ -z "$LAST_LINE" ]]; then
        echo "NO_STEPS"
        return 1
    fi

    CURRENT_STEP=$(echo "$LAST_LINE" | grep -oP 'step\s+\K[0-9]+' || echo "0")
    CURRENT_LOSS=$(echo "$LAST_LINE" | grep -oP 'loss\s+\K[0-9.]+' || echo "N/A")
    CURRENT_LR=$(echo "$LAST_LINE" | grep -oP 'lr\s+\K[0-9.e+-]+' || echo "N/A")
    CURRENT_GNORM=$(echo "$LAST_LINE" | grep -oP 'gnorm\s+\K[0-9.]+' || echo "N/A")
    CURRENT_TOKPS=$(echo "$LAST_LINE" | grep -oP 'tok/s\s+\K[\d,]+' | tr -d ',' || echo "0")
    CURRENT_MEM=$(echo "$LAST_LINE" | grep -oP 'mem\s+\K[0-9.]+GB' || echo "N/A")
    CURRENT_EPOCH=$(echo "$LAST_LINE" | grep -oP 'epoch\s+\K[0-9]+' || echo "0")

    # Log timestamp — parse from the line itself
    LOG_TS=$(echo "$LAST_LINE" | grep -oP '\[\K[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' || echo "unknown")

    return 0
}

# ─── Calculate progress & ETA ─────────────────────────────────────────────────
compute_eta() {
    local step="$1"
    local tokps="$2"

    # Progress
    PROGRESS_PCT=$(echo "scale=1; $step * 100 / $TOTAL_STEPS" | bc -l 2>/dev/null || echo "0")

    # Steps remaining
    STEPS_LEFT=$(( TOTAL_STEPS - step ))

    # Tokens processed so far (approx: step × 2M tokens/step for 3B, bs=4, seqlen=4096, 8gpu)
    # bs=4, accum=8, 8gpu → effective batch = 4*8*8=256 sequences × 4096 tokens = 1,048,576 ≈ 1M tok/step
    TOKENS_PROCESSED_B=$(echo "scale=2; $step * 1048576 / 1000000000" | bc -l 2>/dev/null || echo "0")

    # ETA using current tok/s
    if [[ "$tokps" -gt 0 ]]; then
        # tokens remaining
        local tokens_left_b
        tokens_left_b=$(echo "scale=2; ($TOTAL_STEPS - $step) * 1048576 / 1000000000" | bc -l 2>/dev/null || echo "0")
        local tokens_left
        tokens_left=$(echo "scale=0; ($TOTAL_STEPS - $step) * 1048576" | bc -l 2>/dev/null || echo "0")
        local secs_left
        secs_left=$(echo "scale=0; $tokens_left / $tokps" | bc -l 2>/dev/null || echo "0")

        ETA_HOURS=$(echo "scale=1; $secs_left / 3600" | bc -l 2>/dev/null || echo "N/A")
        if [[ "$ETA_HOURS" != "N/A" ]]; then
            local eta_epoch
            eta_epoch=$(( $(date +%s) + secs_left ))
            ETA_DATETIME=$(date -d "@$eta_epoch" '+%m/%d %H:%M' 2>/dev/null || echo "N/A")
        else
            ETA_DATETIME="N/A"
        fi
    else
        ETA_HOURS="N/A"
        ETA_DATETIME="N/A"
    fi
}

# ─── GPU summary ─────────────────────────────────────────────────────────────
get_gpu_summary() {
    if ! command -v nvidia-smi &>/dev/null; then
        GPU_SUMMARY="nvidia-smi not available"
        GPU_AVG_UTIL="N/A"
        GPU_TOTAL_MEM="N/A"
        return
    fi

    local raw
    raw=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null || echo "")

    if [[ -z "$raw" ]]; then
        GPU_SUMMARY="GPU query failed"
        GPU_AVG_UTIL="N/A"
        GPU_TOTAL_MEM="N/A"
        return
    fi

    # avg util
    GPU_AVG_UTIL=$(echo "$raw" | awk -F', ' '{sum+=$2; count++} END {printf "%.0f%%", sum/count}')

    # total mem used / total
    GPU_TOTAL_MEM=$(echo "$raw" | awk -F', ' \
        '{used+=$3; total+=$4} END {printf "%.1f / %.1f GiB", used/1024, total/1024}')

    # Per-GPU one-liner: "G0:95% 48G | G1:94% 48G | ..."
    GPU_SUMMARY=$(echo "$raw" | awk -F', ' \
        '{printf "G%s:%s%% %sMiB | ", $1, $2, $3}' | sed 's/ | $//')
}

# ─── Disk usage ──────────────────────────────────────────────────────────────
get_disk_info() {
    DISK_INFO=$(df -h "$CKPT_DIR" 2>/dev/null | awk 'NR==2 {printf "%s used / %s total (%s)", $3, $2, $5}' || echo "N/A")
    CKPT_COUNT=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | wc -l || echo "0")
    LAST_CKPT=$(ls -dt "$CKPT_DIR"/checkpoint-* 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "none")
}

# ─── Process status ───────────────────────────────────────────────────────────
get_process_status() {
    PROC_STATUS="UNKNOWN"
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            PROC_STATUS="RUNNING (PID $pid)"
        else
            PROC_STATUS="STOPPED (PID $pid)"
        fi
    else
        PROC_STATUS="NO PID FILE"
    fi
}

# ─── Build & send message ────────────────────────────────────────────────────
build_and_send() {
    local step="$CURRENT_STEP"
    local loss="$CURRENT_LOSS"
    local tokps="$CURRENT_TOKPS"

    # Status icon
    local status_icon
    if [[ "$PROC_STATUS" == RUNNING* ]]; then
        status_icon="&#9989;"    # green check
    else
        status_icon="&#10060;"   # red X
    fi

    # Progress bar (20 chars)
    local bar_filled=$(echo "scale=0; $PROGRESS_PCT * 20 / 100" | bc -l 2>/dev/null || echo "0")
    local bar_empty=$(( 20 - bar_filled ))
    PROGRESS_BAR=$(printf '%0.s&#9608;' $(seq 1 $bar_filled 2>/dev/null) ; printf '%0.s&#9617;' $(seq 1 $bar_empty 2>/dev/null)) || PROGRESS_BAR="[$PROGRESS_PCT%]"

    local msg
    msg="$(cat <<EOF
<b>FRANKENSTALLM 3B — Hourly Status</b>
<i>$(ts)</i>

$status_icon <b>Process:</b> $PROC_STATUS

<b>Progress</b>
Step: <code>$step / $TOTAL_STEPS</code>  ($PROGRESS_PCT%)
Tokens: <code>${TOKENS_PROCESSED_B}B / ${TOTAL_TOKENS_B}B</code>
Epoch: <code>$CURRENT_EPOCH</code>
Last log: <code>$LOG_TS</code>

<b>Training Metrics</b>
Loss:   <code>$loss</code>
LR:     <code>$CURRENT_LR</code>
Gnorm:  <code>$CURRENT_GNORM</code>
Tok/s:  <code>$tokps</code>
Mem:    <code>$CURRENT_MEM</code>

<b>ETA</b>
Steps left: <code>$STEPS_LEFT</code>
Remaining:  <code>~$ETA_HOURS h</code>
Est. done:  <code>$ETA_DATETIME</code>

<b>GPU</b>
Avg util: <code>$GPU_AVG_UTIL</code>
Total mem: <code>$GPU_TOTAL_MEM</code>

<b>Checkpoints</b>
Last saved: <code>$LAST_CKPT</code>
Total: <code>$CKPT_COUNT</code> checkpoints

<b>Disk</b>
<code>$DISK_INFO</code>
EOF
)"

    log "Sending hourly status report (step $step)..."
    $NOTIFY "$msg" || {
        log "ERROR: Failed to send Telegram message."
        return 1
    }
    log "Status report sent."
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    log "=== Hourly status START ==="

    parse_log || {
        log "Cannot parse log — sending minimal status."
        $NOTIFY "<b>FRANKENSTALLM 3B</b> — Status check at $(ts)

<b>WARNING:</b> Cannot read training log at:
<code>$LOG_FILE</code>

Process status: $(cat "$PID_FILE" 2>/dev/null && echo "(PID found)" || echo "(no PID file)")" || true
        return 0
    }

    compute_eta "$CURRENT_STEP" "$CURRENT_TOKPS"
    get_gpu_summary
    get_disk_info
    get_process_status
    build_and_send

    log "=== Hourly status END ==="
}

main "$@"
