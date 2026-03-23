#!/bin/bash
# sft_status.sh — SFT 학습 현황 리포트
# Usage: bash scripts/sft_status.sh

LOG="/root/taketimes/llm/EVAFRILL-Mo/checkpoints/3b_sft_v2/train.log"
TOTAL_STEPS=135000

if [ ! -f "$LOG" ]; then
    echo "[ERROR] Log file not found: $LOG"
    exit 1
fi

# Latest training line (exclude val_loss, shutdown, etc.)
LATEST=$(grep 'tok/s' "$LOG" | tail -1)
STEP=$(echo "$LATEST" | grep -oP 'step\s+\K\d+')
LOSS=$(echo "$LATEST" | grep -oP 'loss \K[\d.]+')
LR=$(echo "$LATEST" | grep -oP 'lr \K[\S]+')
GNORM=$(echo "$LATEST" | grep -oP 'gnorm \K[\d.]+')
TOKS=$(echo "$LATEST" | grep -oP 'tok/s \K[\d,]+')
TIME=$(echo "$LATEST" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')

PCT=$(awk "BEGIN {printf \"%.2f\", $STEP/$TOTAL_STEPS*100}")
REMAIN=$((TOTAL_STEPS - STEP))

# Speed: avg seconds per step from last 100 train log lines (~1000 steps)
# Filter only "step ... loss" lines (exclude val_loss, shutdown, etc.)
FIRST_OF_100=$(grep 'tok/s' "$LOG" | tail -100 | head -1)
STEP_A=$(echo "$FIRST_OF_100" | grep -oP 'step\s+\K\d+')
TIME_A=$(echo "$FIRST_OF_100" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
STEP_B=$STEP
TIME_B=$TIME

EPOCH_A=$(date -d "$TIME_A" +%s 2>/dev/null)
EPOCH_B=$(date -d "$TIME_B" +%s 2>/dev/null)

if [ -n "$EPOCH_A" ] && [ -n "$EPOCH_B" ] && [ "$STEP_B" -gt "$STEP_A" ]; then
    ELAPSED=$((EPOCH_B - EPOCH_A))
    STEPS_DONE=$((STEP_B - STEP_A))
    SEC_PER_STEP=$(awk "BEGIN {printf \"%.1f\", $ELAPSED/$STEPS_DONE}")
    ETA_TRAIN_SEC=$(awk "BEGIN {printf \"%.0f\", $REMAIN * $ELAPSED / $STEPS_DONE}")
    # Validation overhead: ~300s per eval, every 5000 steps
    EVAL_INTERVAL=5000
    VAL_DURATION=92  # 실측: step 5K/10K 모두 ~92초
    REMAINING_EVALS=$(awk "BEGIN {printf \"%.0f\", $REMAIN / $EVAL_INTERVAL}")
    VAL_TOTAL_SEC=$((REMAINING_EVALS * VAL_DURATION))
    ETA_SEC=$((ETA_TRAIN_SEC + VAL_TOTAL_SEC))
    ETA_HOURS=$(awk "BEGIN {printf \"%.1f\", $ETA_SEC/3600}")
    ETA_DAYS=$(awk "BEGIN {printf \"%.1f\", $ETA_SEC/86400}")
    ETA_DATE=$(date -d "+${ETA_SEC} seconds" '+%m/%d %H:%M' 2>/dev/null)
    VAL_HOURS=$(awk "BEGIN {printf \"%.1f\", $VAL_TOTAL_SEC/3600}")
else
    SEC_PER_STEP="?"
    ETA_HOURS="?"
    ETA_DAYS="?"
    ETA_DATE="?"
fi

# Loss trend: 1000 steps ago vs now
LINE_1K=$(grep -oP 'step\s+\K\d+' "$LOG" | awk -v t=$((STEP-1000)) '{d=$1-t; if(d<0)d=-d; if(best=="" || d<bestd){bestd=d;best=$1}}END{print best}')
if [ -n "$LINE_1K" ]; then
    LOSS_1K=$(grep "step *${LINE_1K} " "$LOG" | tail -1 | grep -oP 'loss \K[\d.]+')
else
    LOSS_1K="?"
fi

# Loss trend: 5000 steps ago vs now
LINE_5K=$(grep -oP 'step\s+\K\d+' "$LOG" | awk -v t=$((STEP-5000)) '{d=$1-t; if(d<0)d=-d; if(best=="" || d<bestd){bestd=d;best=$1}}END{print best}')
if [ -n "$LINE_5K" ]; then
    LOSS_5K=$(grep "step *${LINE_5K} " "$LOG" | tail -1 | grep -oP 'loss \K[\d.]+')
else
    LOSS_5K="?"
fi

echo "============================================"
echo "  EVAFRILL-Mo 3B SFT — Status Report"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""
echo "  Step      : $STEP / $TOTAL_STEPS  ($PCT%)"
echo "  Remaining : $REMAIN steps"
echo "  Speed     : ${SEC_PER_STEP}s/step  (avg last ~${STEPS_DONE:-?} steps)"
echo "  ETA       : ${ETA_HOURS}h (${ETA_DAYS} days) → $ETA_DATE"
echo "              (train ${ETA_TRAIN_SEC:-?}s + val ${REMAINING_EVALS:-?}회×5min = +${VAL_HOURS:-?}h)"
echo ""
echo "  Loss (now)     : $LOSS"
echo "  Loss (~1K ago) : $LOSS_1K"
echo "  Loss (~5K ago) : $LOSS_5K"
echo "  Grad norm      : $GNORM"
echo "  LR             : $LR"
echo "  tok/s          : $TOKS"
echo ""
echo "  Last log: $TIME"
echo "============================================"
