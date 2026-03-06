#!/usr/bin/env bash
# =============================================================================
# apply_optimizations.sh — Apply v2 optimizations and restart training
#
# Optimizations applied:
#   1. QKV Projection Fusion (+8-12% throughput)
#   2. NUMA CPU Affinity (fix 69% cross-NUMA workers)
#   3. Batch size 4→5 (11h saved over full run)
#   4. NCCL NVLS algorithm + 256MB buffers
#   5. DDP bucket_cap_mb 400→800
#   6. DataLoader num_workers 4→6, prefetch_factor 3→4
#   7. MADV_RANDOM + WILLNEED for PackedDataset
#   8. numactl --interleave=all on torchrun
#
# Usage:
#   bash scripts/apply_optimizations.sh              # full migration
#   bash scripts/apply_optimizations.sh --test-only  # just validate, don't restart
#   bash scripts/apply_optimizations.sh --skip-stop  # don't stop current training
# =============================================================================
set -u

cd "$(dirname "$0")/.."

RUN_NAME="korean_3b_fp8_run1"
CKPT_DIR="checkpoints/${RUN_NAME}"
PID_FILE="${CKPT_DIR}/train.pid"
LOG_FILE="${CKPT_DIR}/train.log"

TEST_ONLY=false
SKIP_STOP=false
for arg in "$@"; do
    case "$arg" in
        --test-only) TEST_ONLY=true ;;
        --skip-stop) SKIP_STOP=true ;;
    esac
done

echo "=================================================================="
echo "  FRANKENSTALLM 3B — Optimization Migration v2"
echo "  $(date)"
echo "=================================================================="

# ---- Step 1: Validate all modified files --------------------------------
echo ""
echo "[1/6] Validating modified files..."
ERRORS=0

for pyfile in model/attention.py train/pretrain.py data/dataset.py scripts/migrate_qkv_checkpoint.py; do
    if python3 -c "import ast; ast.parse(open('$pyfile').read())" 2>/dev/null; then
        echo "  ✓ $pyfile — syntax OK"
    else
        echo "  ✗ $pyfile — SYNTAX ERROR"
        ((ERRORS++))
    fi
done

if bash -n scripts/launch_3b_pretrain.sh 2>/dev/null; then
    echo "  ✓ scripts/launch_3b_pretrain.sh — syntax OK"
else
    echo "  ✗ scripts/launch_3b_pretrain.sh — SYNTAX ERROR"
    ((ERRORS++))
fi

# Check YAML
python3 -c "
import yaml
with open('configs/korean_3b_fp8.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['train']['batch_size'] == 5, f'batch_size should be 5, got {cfg[\"train\"][\"batch_size\"]}'
print('  ✓ configs/korean_3b_fp8.yaml — valid, batch_size=5')
" 2>/dev/null || { echo "  ✗ configs/korean_3b_fp8.yaml — INVALID"; ((ERRORS++)); }

if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "[ERROR] $ERRORS file(s) failed validation. Aborting."
    exit 1
fi
echo "  All files validated successfully."

if $TEST_ONLY; then
    echo ""
    echo "[INFO] --test-only mode. Exiting without restart."
    exit 0
fi

# ---- Step 2: Stop current training (graceful) ---------------------------
if ! $SKIP_STOP; then
    echo ""
    echo "[2/6] Stopping current training (SIGTERM → emergency checkpoint)..."
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "  Sending SIGTERM to PID $PID..."
            kill "$PID"
            echo "  Waiting for graceful shutdown (up to 120s)..."
            for i in $(seq 1 120); do
                if ! kill -0 "$PID" 2>/dev/null; then
                    echo "  Process stopped after ${i}s"
                    break
                fi
                sleep 1
            done
            if kill -0 "$PID" 2>/dev/null; then
                echo "  [WARN] Process still running after 120s. Force killing..."
                kill -9 "$PID" 2>/dev/null || true
                sleep 2
            fi
        else
            echo "  Process $PID not running."
        fi
    else
        echo "  No PID file found."
    fi

    # Wait for all GPU processes to clear
    echo "  Waiting for GPU processes to terminate..."
    for i in $(seq 1 30); do
        if ! pgrep -f "pretrain.py.*korean_3b" >/dev/null 2>&1; then
            echo "  All GPU processes cleared."
            break
        fi
        sleep 1
    done
fi

# ---- Step 3: Find and migrate latest checkpoint -------------------------
echo ""
echo "[3/6] Migrating latest checkpoint (QKV fusion)..."
LATEST_CKPT=$(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [[ -z "$LATEST_CKPT" ]]; then
    echo "  [ERROR] No checkpoint found!"
    exit 1
fi
echo "  Latest checkpoint: $LATEST_CKPT"

# Backup original model.pt
cp "${LATEST_CKPT}/model.pt" "${LATEST_CKPT}/model.pt.backup_pre_qkv"
echo "  Backup created: model.pt.backup_pre_qkv"

# Run migration
python3 scripts/migrate_qkv_checkpoint.py "$LATEST_CKPT"
echo "  QKV fusion migration complete."

# ---- Step 4: Quick validation test (5 steps) ----------------------------
echo ""
echo "[4/6] Running 5-step validation test..."
# Use single GPU for fast test
timeout 120 python3 train/pretrain.py \
    --config configs/korean_3b_fp8.yaml \
    --train_data data/3b_train.bin \
    --checkpoint_dir /tmp/frankenstallm_test \
    --max_steps 5 \
    --batch_size 5 \
    --resume "$LATEST_CKPT" \
    2>&1 | tail -10

TEST_EXIT=$?
if [[ $TEST_EXIT -eq 0 ]]; then
    echo "  ✓ 5-step test passed!"
else
    echo "  ✗ 5-step test FAILED (exit code $TEST_EXIT)"
    echo "  [WARN] Restoring original checkpoint..."
    cp "${LATEST_CKPT}/model.pt.backup_pre_qkv" "${LATEST_CKPT}/model.pt"
    echo "  Original checkpoint restored. Aborting."
    exit 1
fi

# ---- Step 5: Clean up test artifacts ------------------------------------
echo ""
echo "[5/6] Cleaning up test artifacts..."
rm -rf /tmp/frankenstallm_test

# ---- Step 6: Launch full training with optimizations --------------------
echo ""
echo "[6/6] Launching optimized training..."
echo ""
echo "  Changes applied:"
echo "    • QKV Projection Fusion (single GEMM)"
echo "    • NUMA CPU Affinity (cores 0-35→GPU0-3, 36-71→GPU4-7)"
echo "    • Batch size: 4 → 5"
echo "    • NCCL: NVLS,Ring algorithm, 256MB buffers"
echo "    • DDP: bucket_cap_mb 400 → 800"
echo "    • DataLoader: 4→6 workers, prefetch 3→4"
echo "    • MADV_RANDOM + WILLNEED for dataset mmap"
echo "    • numactl --interleave=all on torchrun"
echo ""

bash scripts/launch_3b_pretrain.sh

echo ""
echo "=================================================================="
echo "  Migration complete! Monitor with:"
echo "    tail -f ${LOG_FILE}"
echo "=================================================================="
