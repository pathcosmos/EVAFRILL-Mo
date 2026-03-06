#!/usr/bin/env bash
# ============================================================
# run_eval_full.sh — 전체 한국어 벤치마크 평가 (목표: 1.5-3시간)
#
# 사용법:
#   bash scripts/run_eval_full.sh [CHECKPOINT_DIR] [OUTPUT_DIR]
#
# 예시:
#   bash scripts/run_eval_full.sh \
#       checkpoints/korean_1b_sft/checkpoint-0005000 \
#       eval/outputs/full_5000
#
# 태스크:
#   - KoBEST (5): boolq, copa, hellaswag, sentineg, wic
#   - HAE-RAE Bench (5): general_knowledge, history, loan_word, rare_word, standard_nomenclature
#   - Global MMLU Korean: 57개 도메인
#   - PAWS-Ko: 패러프레이즈 탐지
#   - KorMedMCQA: 한국어 의학 MCQ (선택)
#
# 총 예상 샘플: ~15,000개
# 1B 모델 @ 8×B200 기준: 약 1.5-3시간
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ─── 인자 처리 ────────────────────────────────────────────
CHECKPOINT="${1:-checkpoints/korean_1b_sft/checkpoint-0005000}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${2:-eval/outputs/full_${TIMESTAMP}}"

[[ "$CHECKPOINT" != /* ]] && CHECKPOINT="$PROJECT_DIR/$CHECKPOINT"
[[ "$OUTPUT_DIR" != /* ]]  && OUTPUT_DIR="$PROJECT_DIR/$OUTPUT_DIR"

# ─── 설정 ────────────────────────────────────────────────
HF_MODEL_DIR="$PROJECT_DIR/outputs/hf_$(basename "$CHECKPOINT")"
TOKENIZER="$PROJECT_DIR/tokenizer/korean_sp/tokenizer.json"

# GPU 설정: 단일 GPU 또는 tensor parallel
# lm-eval의 hf backend는 기본 단일 GPU 사용
# 멀티 GPU: --model_args "pretrained=...,parallelize=True" (자동 device_map)
USE_MULTI_GPU="${USE_MULTI_GPU:-0}"
if [ "$USE_MULTI_GPU" = "1" ]; then
    MODEL_EXTRA_ARGS=",parallelize=True"
    echo "▶ 멀티 GPU 모드 활성화 (device_map=auto)"
else
    MODEL_EXTRA_ARGS=""
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"

# ─── 태스크 정의 ─────────────────────────────────────────
# Core Korean tasks (항상 실행)
TASKS_CORE="kobest,haerae,paws_ko"

# Extended tasks (시간 있을 때)
TASKS_EXTENDED="global_mmlu_ko"

# 선택적 태스크
TASKS_OPTIONAL="kormedmcqa"    # 한국어 의학 MCQ

# 전체 실행 태스크
TASKS="${TASKS_CORE},${TASKS_EXTENDED}"

# ─── 의존성 확인 ─────────────────────────────────────────
check_dep() {
    python3 -c "import $1" 2>/dev/null || { echo "❌ $1 not found. pip install $2"; exit 1; }
}
check_dep lm_eval lm-eval
check_dep transformers transformers
check_dep safetensors safetensors

echo "=================================================="
echo " Ko-LLM Full Benchmark Evaluation"
echo "=================================================="
echo " Checkpoint  : $CHECKPOINT"
echo " HF output   : $HF_MODEL_DIR"
echo " Tasks       : $TASKS"
echo " Few-shot    : $NUM_FEWSHOT"
echo " Batch size  : $BATCH_SIZE"
echo " Output      : $OUTPUT_DIR"
echo " Multi-GPU   : $USE_MULTI_GPU"
echo " Start time  : $(date)"
echo "=================================================="

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/eval_full.log"

# ─── Step 1: HF 포맷 변환 ───────────────────────────────
echo ""
echo "▶ [1/3] 커스텀 체크포인트 → HF 포맷 변환..."

if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
    python3 "$PROJECT_DIR/scripts/convert_to_hf.py" \
        --checkpoint "$CHECKPOINT" \
        --output "$HF_MODEL_DIR" \
        --tokenizer "$TOKENIZER" \
        2>&1 | tee -a "$LOG_FILE"
    echo "✅ HF 변환 완료: $HF_MODEL_DIR"
else
    echo "  ↳ HF 모델 이미 존재, 변환 스킵: $HF_MODEL_DIR"
fi

# ─── Step 2: 전체 평가 ──────────────────────────────────
echo ""
echo "▶ [2/3] lm-eval 전체 평가 시작..."
echo "  ↳ 로그: $LOG_FILE"
START_TIME=$(date +%s)

if [ "$USE_MULTI_GPU" = "1" ]; then
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=$HF_MODEL_DIR,dtype=float16,parallelize=True" \
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --batch_size "$BATCH_SIZE" \
        --output_path "$OUTPUT_DIR" \
        --log_samples \
        --verbosity INFO \
        2>&1 | tee -a "$LOG_FILE"
else
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=$HF_MODEL_DIR,dtype=float16" \
        --tasks "$TASKS" \
        --num_fewshot "$NUM_FEWSHOT" \
        --batch_size "$BATCH_SIZE" \
        --output_path "$OUTPUT_DIR" \
        --log_samples \
        --verbosity INFO \
        2>&1 | tee -a "$LOG_FILE"
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "✅ 평가 완료! 소요: $((ELAPSED/60))분 $((ELAPSED%60))초"

# ─── Step 3: 결과 요약 리포트 생성 ─────────────────────
echo ""
echo "▶ [3/3] 결과 리포트 생성..."

python3 - "$OUTPUT_DIR" "$CHECKPOINT" <<'PYEOF'
import json, glob, sys, os
from datetime import datetime

output_dir = sys.argv[1]
checkpoint = sys.argv[2] if len(sys.argv) > 2 else "unknown"

results_files = sorted(glob.glob(f"{output_dir}/**/*.json", recursive=True))
results_files = [f for f in results_files if "samples_" not in os.path.basename(f)]

report_lines = [
    f"# Ko-LLM Full Eval Report",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Checkpoint: {checkpoint}",
    "",
]

all_results = {}
for rf in results_files:
    try:
        with open(rf) as f:
            data = json.load(f)
        results = data.get("results", {})
        if results:
            all_results.update(results)
    except Exception:
        pass

# KoBEST 요약
kobest_tasks = [k for k in all_results if k.startswith("kobest_")]
if kobest_tasks:
    report_lines.append("## KoBEST")
    report_lines.append("| Task | Metric | Score |")
    report_lines.append("|------|--------|-------|")
    for task in sorted(kobest_tasks):
        metrics = all_results[task]
        for key, val in metrics.items():
            if "stderr" not in key and isinstance(val, (int, float)):
                report_lines.append(f"| {task} | {key} | {val:.4f} |")

# HAE-RAE 요약
haerae_tasks = [k for k in all_results if k.startswith("haerae")]
if haerae_tasks:
    report_lines.append("\n## HAE-RAE Bench")
    report_lines.append("| Task | Metric | Score |")
    report_lines.append("|------|--------|-------|")
    for task in sorted(haerae_tasks):
        metrics = all_results[task]
        for key, val in metrics.items():
            if "stderr" not in key and isinstance(val, (int, float)):
                report_lines.append(f"| {task} | {key} | {val:.4f} |")

# MMLU Ko 요약 (상위 레벨만)
mmlu_top = {k: v for k, v in all_results.items() 
            if k.startswith("global_mmlu_ko") and "_" not in k.replace("global_mmlu_ko", "")}
if mmlu_top:
    report_lines.append("\n## Global MMLU (Korean)")
    for task, metrics in mmlu_top.items():
        for key, val in metrics.items():
            if "stderr" not in key and isinstance(val, (int, float)):
                report_lines.append(f"- {task} {key}: {val:.4f}")

# 기타
other_tasks = [k for k in all_results 
               if not k.startswith("kobest_") 
               and not k.startswith("haerae")
               and not k.startswith("global_mmlu_ko")]
if other_tasks:
    report_lines.append("\n## 기타 태스크")
    for task in sorted(other_tasks):
        metrics = all_results[task]
        for key, val in metrics.items():
            if "stderr" not in key and isinstance(val, (int, float)):
                report_lines.append(f"- {task} | {key}: {val:.4f}")

report_path = os.path.join(output_dir, "SUMMARY.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print("\n".join(report_lines))
print(f"\n📄 리포트 저장: {report_path}")
PYEOF

echo ""
echo "=================================================="
echo "✅ 전체 평가 완료!"
echo " 결과 디렉토리: $OUTPUT_DIR"
echo " 요약 리포트  : $OUTPUT_DIR/SUMMARY.md"
echo " 전체 로그    : $LOG_FILE"
echo " 완료 시각    : $(date)"
echo "=================================================="
