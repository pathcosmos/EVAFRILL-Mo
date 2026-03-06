#!/usr/bin/env bash
# =============================================================================
# deploy_3b_ollama.sh — 3B GGUF 모델을 Ollama에 등록 & 자동 테스트
#
# Usage:
#   bash scripts/deploy_3b_ollama.sh [model_name]
#
#   model_name: Ollama 모델 이름 (default: frankenstallm-3b)
#
# 전제 조건:
#   - ollama 설치: https://ollama.com/download
#   - bash scripts/convert_3b_gguf.sh 실행 완료
#   - outputs/gguf/frankenstallm-3b-Q4_K_M.gguf 존재
#   - Modelfile.3b 존재
# =============================================================================
set -euo pipefail

MODEL_NAME="${1:-frankenstallm-3b}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELFILE="$PROJECT_DIR/Modelfile.3b"
GGUF_PATH="$PROJECT_DIR/outputs/gguf/frankenstallm-3b-Q4_K_M.gguf"

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------
if ! command -v ollama &> /dev/null; then
    echo "ERROR: ollama가 설치되어 있지 않습니다."
    echo "설치: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

if [[ ! -f "$GGUF_PATH" ]]; then
    echo "ERROR: GGUF 파일을 찾을 수 없습니다: $GGUF_PATH"
    echo "먼저 실행: bash scripts/convert_3b_gguf.sh"
    exit 1
fi

if [[ ! -f "$MODELFILE" ]]; then
    echo "ERROR: Modelfile.3b 를 찾을 수 없습니다: $MODELFILE"
    echo "  프로젝트 루트에 Modelfile.3b 가 있어야 합니다."
    exit 1
fi

echo "=================================================================="
echo "  3B 모델 Ollama 배포"
echo "  모델명   : $MODEL_NAME"
echo "  GGUF     : $(du -sh "$GGUF_PATH" | cut -f1)  ($GGUF_PATH)"
echo "  Modelfile: $MODELFILE"
echo "=================================================================="
echo ""

# ---------------------------------------------------------------------------
# Ollama 서버 실행 확인
# ---------------------------------------------------------------------------
if ! ollama list &>/dev/null; then
    echo "[WARN] Ollama 서버가 응답하지 않습니다. 백그라운드로 시작합니다 ..."
    ollama serve &>/tmp/ollama_serve.log &
    OLLAMA_PID=$!
    echo "  PID: $OLLAMA_PID  (로그: /tmp/ollama_serve.log)"
    # 서버 준비 대기 (최대 15초)
    for i in $(seq 1 15); do
        if ollama list &>/dev/null 2>&1; then
            echo "  [OK] Ollama 서버 준비 완료 (${i}초)"
            break
        fi
        sleep 1
    done
fi

# ---------------------------------------------------------------------------
# Ollama 모델 등록
# ---------------------------------------------------------------------------
echo "[1/2] Ollama 모델 등록 중: $MODEL_NAME ..."
ollama create "$MODEL_NAME" -f "$MODELFILE"
echo "  [OK] 등록 완료"

# ---------------------------------------------------------------------------
# 자동 테스트 프롬프트 5개 실행
# ---------------------------------------------------------------------------
echo ""
echo "[2/2] 자동 테스트 프롬프트 실행 (5개) ..."
echo ""

declare -a TEST_PROMPTS=(
    "안녕하세요! 간단히 자기소개를 해주세요."
    "대한민국의 수도는 어디인가요? 그 도시의 특징을 설명해주세요."
    "파이썬으로 피보나치 수열을 출력하는 함수를 작성해주세요."
    "인공지능이 사회에 미치는 긍정적인 영향 3가지를 설명해주세요."
    "오늘 저녁 메뉴로 무엇을 추천해주시겠어요? 이유도 함께 말씀해주세요."
)

PASS_COUNT=0
FAIL_COUNT=0
TOTAL=${#TEST_PROMPTS[@]}

for i in "${!TEST_PROMPTS[@]}"; do
    PROMPT="${TEST_PROMPTS[$i]}"
    NUM=$((i + 1))
    echo "--- 테스트 $NUM/$TOTAL ---"
    echo "프롬프트: $PROMPT"
    echo ""

    # ollama run: 타임아웃 60초, 응답 첫 300자만 표시
    if RESPONSE=$(timeout 60 ollama run "$MODEL_NAME" "$PROMPT" 2>&1); then
        RESP_PREVIEW="${RESPONSE:0:300}"
        echo "응답: $RESP_PREVIEW"
        if [[ ${#RESPONSE} -gt 300 ]]; then
            echo "      ... (총 ${#RESPONSE}자)"
        fi
        echo "[OK] 테스트 $NUM 성공"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        EXIT_CODE=$?
        echo "[FAIL] 테스트 $NUM 실패 (exit code: $EXIT_CODE)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# 결과 요약
# ---------------------------------------------------------------------------
echo "=================================================================="
echo "  배포 & 테스트 완료"
echo ""
echo "  모델명  : $MODEL_NAME"
echo "  테스트  : $PASS_COUNT/$TOTAL 성공  ($FAIL_COUNT 실패)"
echo ""
if [[ $FAIL_COUNT -eq 0 ]]; then
    echo "  [PASS] 모든 테스트 통과"
else
    echo "  [WARN] 일부 테스트 실패 — 로그를 확인하세요"
fi
echo ""
echo "  Ollama 사용법:"
echo "    ollama run $MODEL_NAME"
echo "    ollama run $MODEL_NAME '질문을 여기에 입력하세요'"
echo "    ollama rm $MODEL_NAME   (삭제)"
echo ""
echo "  Quality Gate:"
echo "    bash scripts/quality_gate.sh deploy"
echo "=================================================================="

[[ $FAIL_COUNT -gt 0 ]] && exit 1 || exit 0
