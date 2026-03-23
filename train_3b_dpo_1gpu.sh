#!/usr/bin/env bash
# ==========================================================================
# EVAFRILL-Mo 3B DPO Training — H100 MIG 3g.40gb (Single GPU)
#
# Runs DPO Round 1 + Round 2 sequentially with resilient restart.
#
# Round 1: Full preference data, 3000 steps, beta=0.1, lr=5e-7
# Round 2: High-quality subset, 2000 steps, beta=0.05, lr=1e-7
# ==========================================================================

set -euo pipefail

# Activate Python env if needed
if [ -f /root/ai-env/bin/activate ]; then
    source /root/ai-env/bin/activate
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SFT_CKPT="checkpoints/3b_sft_v2/checkpoint-best"
DPO_DATA="data/preference/combined_preference.jsonl"
TOKENIZER="tokenizer/korean_sp/tokenizer.json"

# ==========================================
# DPO Round 1
# ==========================================
echo "=========================================="
echo "DPO Round 1: Full data, 3000 steps"
echo "=========================================="

ROUND1_DIR="checkpoints/3b_dpo_r1"
ROUND1_LOG="checkpoints/3b_dpo_r1/train.log"

MAX_RETRIES=5
RETRY_DELAY=30
RETRIES=0

while [ $RETRIES -lt $MAX_RETRIES ]; do
    # Find latest checkpoint for resume
    RESUME_ARG=""
    if [ -d "$ROUND1_DIR" ]; then
        LATEST=$(ls -td "$ROUND1_DIR"/checkpoint-[0-9]* 2>/dev/null | head -1)
        if [ -n "$LATEST" ] && [ -f "$LATEST/model.pt" ]; then
            RESUME_ARG="--resume $LATEST"
            echo "Resuming from: $LATEST"
        fi
    fi

    python3 train/dpo.py \
        --sft_checkpoint "$SFT_CKPT" \
        --dpo_data "$DPO_DATA" \
        --tokenizer "$TOKENIZER" \
        --checkpoint_dir "$ROUND1_DIR" \
        --config configs/h100_mig/dpo_3b_1gpu.yaml \
        --device cuda:0 \
        --log_file "$ROUND1_LOG" \
        $RESUME_ARG \
        && break

    RETRIES=$((RETRIES + 1))
    echo "[WARN] Round 1 crashed (attempt $RETRIES/$MAX_RETRIES). Restarting in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

if [ $RETRIES -ge $MAX_RETRIES ]; then
    echo "[ERROR] Round 1 failed after $MAX_RETRIES attempts"
    exit 1
fi

echo "Round 1 complete."

# ==========================================
# DPO Round 2 (more conservative)
# ==========================================
echo ""
echo "=========================================="
echo "DPO Round 2: Conservative, 2000 steps"
echo "=========================================="

# Use Round 1 merged checkpoint as base
ROUND1_MERGED="$ROUND1_DIR/checkpoint-merged"
if [ ! -d "$ROUND1_MERGED" ]; then
    echo "[ERROR] Round 1 merged checkpoint not found: $ROUND1_MERGED"
    exit 1
fi

ROUND2_DIR="checkpoints/3b_dpo_r2"
ROUND2_LOG="checkpoints/3b_dpo_r2/train.log"
RETRIES=0

while [ $RETRIES -lt $MAX_RETRIES ]; do
    RESUME_ARG=""
    if [ -d "$ROUND2_DIR" ]; then
        LATEST=$(ls -td "$ROUND2_DIR"/checkpoint-[0-9]* 2>/dev/null | head -1)
        if [ -n "$LATEST" ] && [ -f "$LATEST/model.pt" ]; then
            RESUME_ARG="--resume $LATEST"
            echo "Resuming from: $LATEST"
        fi
    fi

    python3 train/dpo.py \
        --sft_checkpoint "$ROUND1_MERGED" \
        --dpo_data "$DPO_DATA" \
        --tokenizer "$TOKENIZER" \
        --checkpoint_dir "$ROUND2_DIR" \
        --max_steps 2000 \
        --beta 0.05 \
        --lr 1e-7 \
        --warmup_steps 50 \
        --device cuda:0 \
        --log_file "$ROUND2_LOG" \
        $RESUME_ARG \
        && break

    RETRIES=$((RETRIES + 1))
    echo "[WARN] Round 2 crashed (attempt $RETRIES/$MAX_RETRIES). Restarting in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
done

if [ $RETRIES -ge $MAX_RETRIES ]; then
    echo "[ERROR] Round 2 failed after $MAX_RETRIES attempts"
    exit 1
fi

echo "Round 2 complete."

# ==========================================
# SLERP Merge: SFT + DPO
# ==========================================
echo ""
echo "=========================================="
echo "SLERP Merge: SFT ↔ DPO"
echo "=========================================="

ROUND2_MERGED="$ROUND2_DIR/checkpoint-merged"
if [ ! -d "$ROUND2_MERGED" ]; then
    echo "[WARN] Round 2 merged checkpoint not found, using Round 1"
    ROUND2_MERGED="$ROUND1_MERGED"
fi

python3 scripts/merge_checkpoints.py \
    --ckpt_a "$SFT_CKPT" \
    --ckpt_b "$ROUND2_MERGED" \
    --output "checkpoints/3b_dpo/checkpoint-slerp" \
    --alpha 0.5

echo ""
echo "=========================================="
echo "DPO Pipeline Complete!"
echo "  Round 1: $ROUND1_DIR"
echo "  Round 2: $ROUND2_DIR"
echo "  SLERP:   checkpoints/3b_dpo/checkpoint-slerp"
echo "=========================================="
