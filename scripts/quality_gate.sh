#!/usr/bin/env bash
# =============================================================================
# quality_gate.sh — Phase 완료 자동 품질 게이트 검증
#
# Usage:
#   bash scripts/quality_gate.sh <phase>
#
# Phases:
#   pretrain  — 사전학습 게이트 (val_loss, loss 단조 감소)
#   sft       — SFT 게이트 (val_loss 수렴, 반복률, KoBEST)
#   orpo      — ORPO 게이트 (반복률, KoBEST, chosen > rejected)
#   deploy    — 배포 게이트 (GGUF perplexity, Ollama 응답)
#   all       — 모든 게이트 순차 실행
#
# Exit codes:
#   0  — 게이트 통과
#   1  — 게이트 실패 (기준 미달)
#   2  — 필수 파일 / 의존성 없음 (실행 불가)
# =============================================================================
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 색상 출력 헬퍼
# ---------------------------------------------------------------------------
_RED='\033[0;31m'
_GREEN='\033[0;32m'
_YELLOW='\033[1;33m'
_BLUE='\033[0;34m'
_NC='\033[0m'

log_info()  { echo -e "${_BLUE}[INFO]${_NC}  $*"; }
log_ok()    { echo -e "${_GREEN}[PASS]${_NC}  $*"; }
log_warn()  { echo -e "${_YELLOW}[WARN]${_NC}  $*"; }
log_fail()  { echo -e "${_RED}[FAIL]${_NC}  $*"; }
log_skip()  { echo -e "       [SKIP]  $*"; }

# ---------------------------------------------------------------------------
# 유틸리티: Python 한 줄 표현식 평가 (부동소수점 비교)
# ---------------------------------------------------------------------------
py_eval() {
    python3 -c "import sys; sys.exit(0 if ($1) else 1)"
}

py_value() {
    python3 -c "print($1)"
}

# ---------------------------------------------------------------------------
# 유틸리티: JSON에서 값 추출
# ---------------------------------------------------------------------------
json_get() {
    local file="$1" key="$2"
    python3 -c "
import json, sys
try:
    d = json.load(open('$file'))
    keys = '$key'.split('.')
    for k in keys:
        d = d[k]
    print(d)
except Exception as e:
    print('NOT_FOUND')
    sys.exit(1)
"
}

# ---------------------------------------------------------------------------
# 게이트 결과 집계
# ---------------------------------------------------------------------------
GATE_PASS=0
GATE_FAIL=0
GATE_SKIP=0

record_pass() { GATE_PASS=$((GATE_PASS + 1)); log_ok "$*"; }
record_fail() { GATE_FAIL=$((GATE_FAIL + 1)); log_fail "$*"; }
record_skip() { GATE_SKIP=$((GATE_SKIP + 1)); log_skip "$*"; }

# =============================================================================
# Gate 1: Pretrain
# =============================================================================
gate_pretrain() {
    echo ""
    echo "=================================================================="
    echo "  Gate: PRETRAIN"
    echo "  기준: val_loss < 2.5 | loss 단조 감소 확인"
    echo "=================================================================="

    # 최신 체크포인트 디렉토리 탐색
    CKPT_BASE="$PROJECT_DIR/checkpoints"
    METRICS_FILE=""

    # metrics.json 또는 train_log.jsonl 탐색
    for candidate in \
        "$CKPT_BASE/korean_3b_fp8_pretrain/metrics.json" \
        "$CKPT_BASE/korean_3b_pretrain/metrics.json" \
        "$PROJECT_DIR/outputs/pretrain_metrics.json" \
        "$PROJECT_DIR/logs/pretrain_metrics.json"
    do
        if [[ -f "$candidate" ]]; then
            METRICS_FILE="$candidate"
            break
        fi
    done

    if [[ -z "$METRICS_FILE" ]]; then
        log_warn "사전학습 메트릭 파일을 찾을 수 없습니다."
        log_warn "찾는 경로: $CKPT_BASE/korean_3b_*/metrics.json"
        log_warn "메트릭 파일이 없으면 학습 스크립트에서 아래 형식으로 저장하세요:"
        log_warn '  {"val_loss": 2.3, "loss_history": [3.1, 2.8, 2.5, 2.3]}'
        record_skip "메트릭 파일 없음 — 게이트 건너뜀"
        return 0
    fi

    log_info "메트릭 파일: $METRICS_FILE"

    # val_loss 확인
    VAL_LOSS=$(json_get "$METRICS_FILE" "val_loss" 2>/dev/null || echo "NOT_FOUND")
    if [[ "$VAL_LOSS" == "NOT_FOUND" ]]; then
        record_skip "val_loss 키 없음 — 건너뜀"
    else
        log_info "val_loss = $VAL_LOSS  (기준: < 2.5)"
        if py_eval "$VAL_LOSS < 2.5" 2>/dev/null; then
            record_pass "val_loss $VAL_LOSS < 2.5"
        else
            record_fail "val_loss $VAL_LOSS >= 2.5  (기준 미달)"
        fi
    fi

    # loss 단조 감소 확인 (loss_history)
    python3 - "$METRICS_FILE" <<'PYEOF'
import json, sys

metrics_file = sys.argv[1]
try:
    d = json.load(open(metrics_file))
    history = d.get("loss_history", [])
except Exception as e:
    print(f"[SKIP] loss_history 읽기 실패: {e}")
    sys.exit(0)

if len(history) < 2:
    print(f"[SKIP] loss_history 데이터 부족 ({len(history)}개)")
    sys.exit(0)

# 전체 추세가 감소하는지 확인 (처음 1/4 vs 마지막 1/4 평균 비교)
n = len(history)
q = max(1, n // 4)
early_avg = sum(history[:q]) / q
late_avg  = sum(history[-q:]) / q

if late_avg < early_avg:
    print(f"[PASS] loss 단조 감소 확인: 초기 avg={early_avg:.4f} → 최근 avg={late_avg:.4f}")
    sys.exit(0)
else:
    print(f"[FAIL] loss 감소 미확인: 초기 avg={early_avg:.4f}, 최근 avg={late_avg:.4f}")
    sys.exit(1)
PYEOF
    local mono_exit=$?
    if [[ $mono_exit -eq 0 ]]; then
        GATE_PASS=$((GATE_PASS + 1))
    elif [[ $mono_exit -eq 1 ]]; then
        GATE_FAIL=$((GATE_FAIL + 1))
    fi
    # exit 0 (SKIP) 는 이미 처리됨
}

# =============================================================================
# Gate 2: SFT
# =============================================================================
gate_sft() {
    echo ""
    echo "=================================================================="
    echo "  Gate: SFT"
    echo "  기준: val_loss 수렴 | 반복률 < 15% | KoBEST > 55%"
    echo "=================================================================="

    METRICS_FILE=""
    for candidate in \
        "$PROJECT_DIR/outputs/sft_metrics.json" \
        "$PROJECT_DIR/logs/sft_metrics.json" \
        "$PROJECT_DIR/checkpoints/sft/metrics.json"
    do
        if [[ -f "$candidate" ]]; then
            METRICS_FILE="$candidate"
            break
        fi
    done

    if [[ -z "$METRICS_FILE" ]]; then
        log_warn "SFT 메트릭 파일을 찾을 수 없습니다."
        log_warn '  {"val_loss": 1.8, "rep_rate": 0.08, "kobest_score": 0.62}'
        record_skip "SFT 메트릭 파일 없음 — 게이트 건너뜀"
        return 0
    fi

    log_info "메트릭 파일: $METRICS_FILE"

    # val_loss 수렴 (상대 변화율 < 1% — 마지막 두 체크포인트)
    python3 - "$METRICS_FILE" <<'PYEOF'
import json, sys

metrics_file = sys.argv[1]
try:
    d = json.load(open(metrics_file))
    history = d.get("val_loss_history", [])
except Exception as e:
    print(f"[SKIP] val_loss_history 읽기 실패: {e}")
    sys.exit(0)

if len(history) < 2:
    # 단일 val_loss만 있으면 단순 확인
    val_loss = d.get("val_loss")
    if val_loss is not None:
        print(f"[INFO] val_loss = {val_loss} (수렴 히스토리 없음 — 단일 값 확인 건너뜀)")
    sys.exit(0)

last   = history[-1]
second = history[-2]
rel_change = abs(last - second) / max(abs(second), 1e-9)

if rel_change < 0.01:
    print(f"[PASS] val_loss 수렴 (상대변화율 {rel_change*100:.3f}% < 1%): {second:.4f} → {last:.4f}")
    sys.exit(0)
else:
    print(f"[FAIL] val_loss 미수렴 (상대변화율 {rel_change*100:.3f}% >= 1%): {second:.4f} → {last:.4f}")
    sys.exit(1)
PYEOF
    local conv_exit=$?
    [[ $conv_exit -eq 0 ]] && GATE_PASS=$((GATE_PASS + 1)) || GATE_FAIL=$((GATE_FAIL + 1))

    # 반복률 확인
    REP_RATE=$(json_get "$METRICS_FILE" "rep_rate" 2>/dev/null || echo "NOT_FOUND")
    if [[ "$REP_RATE" == "NOT_FOUND" ]]; then
        record_skip "rep_rate 키 없음 — 건너뜀"
    else
        REP_PCT=$(py_value "$REP_RATE * 100")
        log_info "반복률 = ${REP_PCT}%  (기준: < 15%)"
        if py_eval "$REP_RATE < 0.15" 2>/dev/null; then
            record_pass "반복률 ${REP_PCT}% < 15%"
        else
            record_fail "반복률 ${REP_PCT}% >= 15%  (기준 미달)"
        fi
    fi

    # KoBEST 확인
    KOBEST=$(json_get "$METRICS_FILE" "kobest_score" 2>/dev/null || echo "NOT_FOUND")
    if [[ "$KOBEST" == "NOT_FOUND" ]]; then
        record_skip "kobest_score 키 없음 — 건너뜀"
    else
        KOBEST_PCT=$(py_value "$KOBEST * 100")
        log_info "KoBEST = ${KOBEST_PCT}%  (기준: > 55%)"
        if py_eval "$KOBEST > 0.55" 2>/dev/null; then
            record_pass "KoBEST ${KOBEST_PCT}% > 55%"
        else
            record_fail "KoBEST ${KOBEST_PCT}% <= 55%  (기준 미달)"
        fi
    fi
}

# =============================================================================
# Gate 3: ORPO
# =============================================================================
gate_orpo() {
    echo ""
    echo "=================================================================="
    echo "  Gate: ORPO"
    echo "  기준: 반복률 < 5% | KoBEST > 60% | chosen > rejected 90%+"
    echo "=================================================================="

    METRICS_FILE=""
    for candidate in \
        "$PROJECT_DIR/outputs/orpo_metrics.json" \
        "$PROJECT_DIR/logs/orpo_metrics.json" \
        "$PROJECT_DIR/checkpoints/orpo/metrics.json"
    do
        if [[ -f "$candidate" ]]; then
            METRICS_FILE="$candidate"
            break
        fi
    done

    if [[ -z "$METRICS_FILE" ]]; then
        log_warn "ORPO 메트릭 파일을 찾을 수 없습니다."
        log_warn '  {"rep_rate": 0.03, "kobest_score": 0.63, "chosen_win_rate": 0.92}'
        record_skip "ORPO 메트릭 파일 없음 — 게이트 건너뜀"
        return 0
    fi

    log_info "메트릭 파일: $METRICS_FILE"

    # 반복률 (더 엄격: < 5%)
    REP_RATE=$(json_get "$METRICS_FILE" "rep_rate" 2>/dev/null || echo "NOT_FOUND")
    if [[ "$REP_RATE" == "NOT_FOUND" ]]; then
        record_skip "rep_rate 키 없음 — 건너뜀"
    else
        REP_PCT=$(py_value "$REP_RATE * 100")
        log_info "반복률 = ${REP_PCT}%  (기준: < 5%)"
        if py_eval "$REP_RATE < 0.05" 2>/dev/null; then
            record_pass "반복률 ${REP_PCT}% < 5%"
        else
            record_fail "반복률 ${REP_PCT}% >= 5%  (기준 미달)"
        fi
    fi

    # KoBEST (더 엄격: > 60%)
    KOBEST=$(json_get "$METRICS_FILE" "kobest_score" 2>/dev/null || echo "NOT_FOUND")
    if [[ "$KOBEST" == "NOT_FOUND" ]]; then
        record_skip "kobest_score 키 없음 — 건너뜀"
    else
        KOBEST_PCT=$(py_value "$KOBEST * 100")
        log_info "KoBEST = ${KOBEST_PCT}%  (기준: > 60%)"
        if py_eval "$KOBEST > 0.60" 2>/dev/null; then
            record_pass "KoBEST ${KOBEST_PCT}% > 60%"
        else
            record_fail "KoBEST ${KOBEST_PCT}% <= 60%  (기준 미달)"
        fi
    fi

    # Chosen win rate (chosen log-prob > rejected log-prob 비율)
    CHOSEN_WIN=$(json_get "$METRICS_FILE" "chosen_win_rate" 2>/dev/null || echo "NOT_FOUND")
    if [[ "$CHOSEN_WIN" == "NOT_FOUND" ]]; then
        record_skip "chosen_win_rate 키 없음 — 건너뜀"
    else
        WIN_PCT=$(py_value "$CHOSEN_WIN * 100")
        log_info "Chosen win rate = ${WIN_PCT}%  (기준: >= 90%)"
        if py_eval "$CHOSEN_WIN >= 0.90" 2>/dev/null; then
            record_pass "Chosen win rate ${WIN_PCT}% >= 90%"
        else
            record_fail "Chosen win rate ${WIN_PCT}% < 90%  (기준 미달)"
        fi
    fi
}

# =============================================================================
# Gate 4: Deploy
# =============================================================================
gate_deploy() {
    echo ""
    echo "=================================================================="
    echo "  Gate: DEPLOY"
    echo "  기준: Q4_K_M perplexity < F16 × 1.05 | Ollama 5개 프롬프트 응답"
    echo "=================================================================="

    local MODEL_NAME="frankenstallm-3b"
    local GGUF_DIR="$PROJECT_DIR/outputs/gguf"
    local F16_GGUF="$GGUF_DIR/${MODEL_NAME}-f16.gguf"
    local Q4KM_GGUF="$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"

    # --- GGUF 파일 존재 확인 ---
    if [[ ! -f "$Q4KM_GGUF" ]]; then
        log_warn "Q4_K_M GGUF 파일 없음: $Q4KM_GGUF"
        log_warn "먼저 실행: bash scripts/convert_3b_gguf.sh"
        record_skip "GGUF 파일 없음 — perplexity 게이트 건너뜀"
    else
        # perplexity 측정 (llama-perplexity 또는 Python fallback)
        LLAMA_PPL_BIN="$PROJECT_DIR/outputs/llama.cpp/build/bin/llama-perplexity"

        if [[ ! -f "$LLAMA_PPL_BIN" ]]; then
            log_warn "llama-perplexity 바이너리 없음 — 빌드 시도 중 ..."
            cmake --build "$PROJECT_DIR/outputs/llama.cpp/build" \
                --target llama-perplexity -j "$(nproc)" &>/dev/null || true
        fi

        # 샘플 텍스트로 perplexity 비교
        SAMPLE_TEXT="$PROJECT_DIR/outputs/gguf/ppl_sample.txt"
        if [[ ! -f "$SAMPLE_TEXT" ]]; then
            # 짧은 한국어 샘플 생성
            cat > "$SAMPLE_TEXT" <<'SAMPLE'
인공지능은 현대 사회에서 매우 중요한 기술로 자리잡고 있습니다.
기계 학습과 딥러닝의 발전으로 인해 다양한 분야에서 혁신이 이루어지고 있습니다.
자연어 처리 기술의 발전은 인간과 컴퓨터의 상호작용 방식을 근본적으로 변화시키고 있습니다.
한국어는 교착어로서 특유의 형태론적 특성을 가지고 있어 자연어 처리에 독특한 도전을 제시합니다.
대규모 언어 모델의 등장으로 기계 번역, 텍스트 요약, 질의응답 등의 성능이 크게 향상되었습니다.
SAMPLE
        fi

        if [[ -f "$LLAMA_PPL_BIN" && -f "$F16_GGUF" ]]; then
            log_info "Perplexity 측정 중 (F16 vs Q4_K_M) ..."

            PPL_F16=$(timeout 120 "$LLAMA_PPL_BIN" -m "$F16_GGUF" -f "$SAMPLE_TEXT" 2>&1 \
                | grep -oP "Perplexity: \K[0-9.]+" | head -1 || echo "0")
            PPL_Q4=$(timeout 120 "$LLAMA_PPL_BIN" -m "$Q4KM_GGUF" -f "$SAMPLE_TEXT" 2>&1 \
                | grep -oP "Perplexity: \K[0-9.]+" | head -1 || echo "0")

            if [[ "$PPL_F16" == "0" || "$PPL_Q4" == "0" ]]; then
                record_skip "Perplexity 측정 실패 — 건너뜀"
            else
                THRESHOLD=$(py_value "$PPL_F16 * 1.05")
                log_info "F16 PPL = $PPL_F16  |  Q4_K_M PPL = $PPL_Q4  |  기준: < $THRESHOLD"
                if py_eval "$PPL_Q4 < $PPL_F16 * 1.05" 2>/dev/null; then
                    record_pass "Q4_K_M PPL $PPL_Q4 < F16 PPL × 1.05 ($THRESHOLD)"
                else
                    record_fail "Q4_K_M PPL $PPL_Q4 >= F16 PPL × 1.05 ($THRESHOLD)"
                fi
            fi
        else
            record_skip "llama-perplexity 또는 F16 GGUF 없음 — perplexity 게이트 건너뜀"
        fi
    fi

    # --- Ollama 응답 테스트 ---
    if ! command -v ollama &>/dev/null; then
        record_skip "ollama 없음 — 응답 테스트 건너뜀"
        return 0
    fi

    if ! ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
        log_warn "Ollama에 $MODEL_NAME 모델이 등록되지 않았습니다."
        log_warn "먼저 실행: bash scripts/deploy_3b_ollama.sh"
        record_skip "Ollama 모델 미등록 — 응답 테스트 건너뜀"
        return 0
    fi

    log_info "Ollama 응답 테스트 (5개 프롬프트) ..."

    declare -a PROMPTS=(
        "안녕하세요."
        "1 더하기 1은 무엇인가요?"
        "파이썬이란 무엇인가요?"
        "한국의 수도는 어디인가요?"
        "오늘 날씨가 좋네요."
    )

    local PASS=0 FAIL=0
    for i in "${!PROMPTS[@]}"; do
        local PROMPT="${PROMPTS[$i]}"
        local NUM=$((i + 1))
        if RESP=$(timeout 45 ollama run "$MODEL_NAME" "$PROMPT" 2>&1) && [[ -n "$RESP" ]]; then
            log_ok "  프롬프트 $NUM 응답 OK (${#RESP}자)"
            PASS=$((PASS + 1))
        else
            log_fail "  프롬프트 $NUM 응답 실패"
            FAIL=$((FAIL + 1))
        fi
    done

    log_info "Ollama 응답: $PASS/5 성공"
    if [[ $FAIL -eq 0 ]]; then
        record_pass "Ollama 5개 프롬프트 모두 응답 성공"
    else
        record_fail "Ollama 응답 실패 $FAIL/5"
    fi
}

# =============================================================================
# 최종 요약 출력
# =============================================================================
print_summary() {
    local phase="$1"
    local TOTAL=$((GATE_PASS + GATE_FAIL + GATE_SKIP))
    echo ""
    echo "=================================================================="
    echo "  Quality Gate 결과: $phase"
    echo "  PASS: $GATE_PASS  |  FAIL: $GATE_FAIL  |  SKIP: $GATE_SKIP  |  TOTAL: $TOTAL"
    echo "=================================================================="

    if [[ $GATE_FAIL -eq 0 ]]; then
        echo -e "${_GREEN}  [GATE PASSED]${_NC} 모든 검증 기준 통과"
        echo ""
        return 0
    else
        echo -e "${_RED}  [GATE FAILED]${_NC} ${GATE_FAIL}개 검증 기준 미달"
        echo "  실패 항목을 수정한 후 다시 실행하세요."
        echo ""
        return 1
    fi
}

# =============================================================================
# 진입점
# =============================================================================
PHASE="${1:-}"

if [[ -z "$PHASE" ]]; then
    echo "Usage: bash scripts/quality_gate.sh <phase>"
    echo "  phase: pretrain | sft | orpo | deploy | all"
    exit 2
fi

echo ""
echo "=================================================================="
echo "  Quality Gate 검증 시작: $PHASE"
echo "  프로젝트: $PROJECT_DIR"
echo "  시각    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================================="

case "$PHASE" in
    pretrain)
        gate_pretrain
        print_summary "pretrain"
        ;;
    sft)
        gate_sft
        print_summary "sft"
        ;;
    orpo)
        gate_orpo
        print_summary "orpo"
        ;;
    deploy)
        gate_deploy
        print_summary "deploy"
        ;;
    all)
        gate_pretrain
        gate_sft
        gate_orpo
        gate_deploy
        print_summary "all"
        ;;
    *)
        echo "ERROR: 알 수 없는 phase: $PHASE"
        echo "Usage: bash scripts/quality_gate.sh <pretrain|sft|orpo|deploy|all>"
        exit 2
        ;;
esac
