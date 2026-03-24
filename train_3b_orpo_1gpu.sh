#!/usr/bin/env bash
# ==========================================================================
# EVAFRILL-Mo 3B ORPO Training — H100 MIG 3g.40gb (Single GPU)
#
# Runs ORPO (Odds Ratio Preference Optimization) with LoRA.
# Single stage: SFT + preference alignment simultaneously.
#
# Base: checkpoints/3b_final/checkpoint-0319772 (pretrained)
# Data: data/preference/combined_preference.jsonl
# ==========================================================================

set -euo pipefail

# Activate Python env if needed
if [ -f /root/ai-env/bin/activate ]; then
    source /root/ai-env/bin/activate
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PRETRAINED_CKPT="checkpoints/3b_final/checkpoint-0319772"
ORPO_DATA="data/preference/combined_preference.jsonl"
TOKENIZER="tokenizer/korean_sp/tokenizer.json"

# ==========================================
# ORPO Training
# ==========================================
echo "=========================================="
echo "ORPO: Single-stage SFT + Preference Align"
echo "=========================================="

ORPO_DIR="checkpoints/3b_orpo"
ORPO_LOG="checkpoints/3b_orpo/train.log"

MAX_RETRIES=5
RETRY_DELAY=30
RETRIES=0

while [ $RETRIES -lt $MAX_RETRIES ]; do
    # Find latest checkpoint for resume
    RESUME_ARG=""
    if [ -d "$ORPO_DIR" ]; then
        LATEST=$(ls -td "$ORPO_DIR"/checkpoint-[0-9]* 2>/dev/null | head -1)
        if [ -n "$LATEST" ] && [ -f "$LATEST/model.pt" ]; then
            RESUME_ARG="--resume $LATEST"
            echo "Resuming from: $LATEST"
        fi
    fi

    python3 train/orpo_native.py \
        --pretrained_checkpoint "$PRETRAINED_CKPT" \
        --orpo_data "$ORPO_DATA" \
        --tokenizer "$TOKENIZER" \
        --checkpoint_dir "$ORPO_DIR" \
        --config configs/h100_mig/orpo_3b_1gpu.yaml \
        --device cuda:0 \
        --log_file "$ORPO_LOG" \
        $RESUME_ARG \
        && break

    RETRIES=$((RETRIES + 1))
    echo "[WARN] ORPO crashed (attempt $RETRIES/$MAX_RETRIES). Restarting in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

if [ $RETRIES -ge $MAX_RETRIES ]; then
    echo "[ERROR] ORPO failed after $MAX_RETRIES attempts"
    exit 1
fi

echo "ORPO training complete."

# ==========================================
# SLERP Merge: SFT + ORPO
# ==========================================
echo ""
echo "=========================================="
echo "SLERP Merge: SFT ↔ ORPO"
echo "=========================================="

SFT_CKPT="checkpoints/3b_sft_v2/checkpoint-best"
ORPO_MERGED="$ORPO_DIR/checkpoint-merged"

if [ ! -d "$ORPO_MERGED" ]; then
    echo "[WARN] ORPO merged checkpoint not found, using final checkpoint"
    LATEST=$(ls -td "$ORPO_DIR"/checkpoint-[0-9]* 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        ORPO_MERGED="$LATEST"
    else
        echo "[ERROR] No ORPO checkpoint found"
        exit 1
    fi
fi

python3 scripts/merge_checkpoints.py \
    --ckpt_a "$SFT_CKPT" \
    --ckpt_b "$ORPO_MERGED" \
    --output "checkpoints/3b_orpo/checkpoint-slerp" \
    --alpha 0.5

echo ""
echo "=========================================="
echo "ORPO Pipeline Complete!"
echo "  ORPO Training: $ORPO_DIR"
echo "  SLERP Merge:   checkpoints/3b_orpo/checkpoint-slerp"
echo "=========================================="
