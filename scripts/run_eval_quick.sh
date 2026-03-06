#!/usr/bin/env bash
# ============================================================
# run_eval_quick.sh — 빠른 평가 체크 (목표: 20-30분)
#
# 사용법:
#   bash scripts/run_eval_quick.sh [CHECKPOINT_DIR] [OUTPUT_DIR]
#
# 예시:
#   bash scripts/run_eval_quick.sh \
#       checkpoints/korean_1b_sft/checkpoint-0005000 \
#       eval/outputs/quick_5000
#
# 태스크: kobest_boolq, kobest_copa, haerae_general_knowledge,
#         haerae_history, paws_ko
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ─── 인자 처리 ────────────────────────────────────────────
CHECKPOINT="${1:-checkpoints/korean_1b_sft/checkpoint-0005000}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${2:-eval/outputs/quick_${TIMESTAMP}}"

# 상대 경로 → 절대 경로
[[ "$CHECKPOINT" != /* ]] && CHECKPOINT="$PROJECT_DIR/$CHECKPOINT"
[[ "$OUTPUT_DIR" != /* ]]  && OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"

# ─── 설정 ────────────────────────────────────────────────
HF_MODEL_DIR="$PROJECT_DIR/outputs/hf_$(basename "$CHECKPOINT")"
TOKENIZER="$PROJECT_DIR/tokenizer/korean_sp/tokenizer.json"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"   # 기본: GPU 0번만 사용
BATCH_SIZE="auto"

# 빠른 체크 태스크 (약 2,000 샘플, ~20분)
TASKS="kobest_boolq,kobest_copa,haerae_general_knowledge,haerae_history,paws_ko"

# ─── 의존성 확인 ─────────────────────────────────────────
check_dep() {
    python3 -c "import $1" 2>/dev/null || { echo "❌ $1 not found. pip install $2"; exit 1; }
}
check_dep lm_eval lm-eval
check_dep transformers transformers
check_dep safetensors safetensors

echo "=================================================="
echo " Ko-LLM Quick Eval"
echo "=================================================="
echo " Checkpoint : $CHECKPOINT"
echo " HF output  : $HF_MODEL_DIR"
echo " Tasks      : $TASKS"
echo " Output     : $OUTPUT_DIR"
echo " Device     : cuda:$DEVICE"
echo "=================================================="

mkdir -p "$OUTPUT_DIR"

# ─── Step 1: HF 포맷 변환 ───────────────────────────────
if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
    echo ""
    echo "▶ Step 1: 커스텀 체크포인트 → HF 포맷 변환..."
    python3 "$PROJECT_DIR/scripts/convert_to_hf.py" \
        --checkpoint "$CHECKPOINT" \
        --output "$HF_MODEL_DIR" \
        --tokenizer "$TOKENIZER"
    echo "✅ HF 변환 완료: $HF_MODEL_DIR"
else
    echo "▶ Step 1: HF 모델 이미 존재, 변환 스킵"
    echo "   $HF_MODEL_DIR"
fi

# ─── Step 2: lm-eval 실행 ───────────────────────────────
echo ""
echo "▶ Step 2: lm-eval 평가 시작..."
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES="$DEVICE" python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$HF_MODEL_DIR,dtype=float16" \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR" \
    --log_samples \
    --verbosity INFO \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "=================================================="
echo "✅ 평가 완료!"
echo " 소요시간: $((ELAPSED / 60))분 $((ELAPSED % 60))초"
echo " 결과 저장: $OUTPUT_DIR"
echo "=================================================="

# ─── Step 3: 결과 요약 출력 ─────────────────────────────
echo ""
echo "▶ Step 3: 결과 요약"
python3 - <<'PYEOF'
import json, glob, sys, os

output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
results_files = glob.glob(f"{output_dir}/**/*.json", recursive=True)
results_files = [f for f in results_files if "results" in f.lower()]

if not results_files:
    print("결과 JSON 파일 없음. eval.log 확인하세요.")
    sys.exit(0)

for rf in results_files:
    try:
        with open(rf) as f:
            data = json.load(f)
        results = data.get("results", {})
        print(f"\n{'='*50}")
        print(f"Task Results (from {os.path.basename(rf)})")
        print(f"{'='*50}")
        for task, metrics in results.items():
            print(f"\n{task}:")
            for key, val in metrics.items():
                if "stderr" not in key and isinstance(val, (int, float)):
                    print(f"  {key}: {val:.4f}")
    except Exception as e:
        print(f"파싱 실패: {rf}: {e}")
PYEOF
python3 - "$OUTPUT_DIR" <<'PYEOF'
import json, glob, sys, os
output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
results_files = glob.glob(f"{output_dir}/**/*.json", recursive=True)
results_files = [f for f in results_files if "results" in os.path.basename(f)]
if not results_files:
    # try finding any json
    results_files = glob.glob(f"{output_dir}/*.json")
for rf in results_files[:3]:
    try:
        with open(rf) as f:
            data = json.load(f)
        results = data.get("results", {})
        print(f"\n{'='*50}\nTask Results: {os.path.basename(rf)}\n{'='*50}")
        for task, metrics in results.items():
            print(f"\n{task}:")
            for key, val in metrics.items():
                if "stderr" not in key and isinstance(val, (int, float)):
                    print(f"  {key}: {val:.4f}")
    except Exception as e:
        print(f"파싱 실패: {rf}: {e}")
PYEOF
