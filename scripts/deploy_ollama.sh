#!/usr/bin/env bash
# =============================================================================
# deploy_ollama.sh — GGUF 모델을 Ollama에 등록
#
# Usage:
#   bash scripts/deploy_ollama.sh [model_name]
#
#   model_name: Ollama 모델 이름 (default: korean-llm-1b)
#
# 전제 조건:
#   - ollama 설치: https://ollama.com/download
#   - bash scripts/convert_to_gguf.sh 실행 완료
#   - outputs/gguf/korean-1b-q4km.gguf 존재
# =============================================================================
set -euo pipefail

MODEL_NAME="${1:-korean-llm-1b}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELFILE="$PROJECT_DIR/Modelfile"
GGUF_PATH="$PROJECT_DIR/outputs/gguf/korean-1b-q4km.gguf"

cd "$PROJECT_DIR"

# --- Pre-flight check -------------------------------------------------------
if ! command -v ollama &> /dev/null; then
    echo "ERROR: ollama not installed."
    echo "Install: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

if [[ ! -f "$GGUF_PATH" ]]; then
    echo "ERROR: GGUF file not found: $GGUF_PATH"
    echo "Run first: bash scripts/convert_to_gguf.sh"
    exit 1
fi

if [[ ! -f "$MODELFILE" ]]; then
    echo "ERROR: Modelfile not found: $MODELFILE"
    exit 1
fi

echo "GGUF 파일: $(du -sh "$GGUF_PATH" | cut -f1)  ($GGUF_PATH)"
echo "Modelfile : $MODELFILE"
echo ""

# --- Register with Ollama ---------------------------------------------------
echo "Ollama 모델 등록 중: $MODEL_NAME ..."
ollama create "$MODEL_NAME" -f "$MODELFILE"

echo ""
echo "=================================================================="
echo "  배포 완료!"
echo "  실행: ollama run $MODEL_NAME"
echo "  예시: ollama run $MODEL_NAME '안녕하세요, 자기소개 해주세요'"
echo "  삭제: ollama rm $MODEL_NAME"
echo "=================================================================="
