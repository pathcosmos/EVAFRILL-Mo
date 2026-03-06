#!/usr/bin/env bash
# =============================================================================
# convert_to_gguf.sh — HuggingFace 포맷 모델을 GGUF로 변환 + Q4_K_M 양자화
#
# Usage:
#   bash scripts/convert_to_gguf.sh [hf_dir] [out_dir]
#
#   hf_dir  : HF 포맷 모델 디렉토리 (default: outputs/hf)
#   out_dir : GGUF 출력 디렉토리    (default: outputs/gguf)
#
# Outputs:
#   outputs/gguf/korean-1b-f16.gguf    — F16 GGUF
#   outputs/gguf/korean-1b-q4km.gguf   — Q4_K_M 양자화 (Ollama용)
#
# 전제 조건:
#   - python scripts/convert_to_hf.py 로 HF 변환 완료
#   - git, cmake, make 설치
#   - pip install safetensors (없으면 pytorch_model.bin으로 fallback)
# =============================================================================
set -euo pipefail

HF_DIR="${1:-outputs/hf}"
OUT_DIR="${2:-outputs/gguf}"
LLAMA_CPP_DIR="outputs/llama.cpp"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

# --- Pre-flight check -------------------------------------------------------
if [[ ! -d "$HF_DIR" ]]; then
    echo "ERROR: HF model directory not found: $HF_DIR"
    echo "Run first: python scripts/convert_to_hf.py --checkpoint <ckpt> --output $HF_DIR"
    exit 1
fi

if [[ ! -f "$HF_DIR/config.json" ]]; then
    echo "ERROR: config.json not found in $HF_DIR"
    exit 1
fi

mkdir -p "$OUT_DIR"

# --- Clone llama.cpp if not present -----------------------------------------
if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    echo "Cloning llama.cpp ..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
fi

# Install Python requirements for conversion script
echo "Installing llama.cpp Python deps ..."
pip install -r "$LLAMA_CPP_DIR/requirements.txt" --break-system-packages -q

# --- Build llama.cpp (for quantization binary) ------------------------------
QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [[ ! -f "$QUANTIZE_BIN" ]]; then
    echo "Building llama.cpp (quantization tool) ..."
    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        2>&1 | tail -5
    cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j "$(nproc)" \
        2>&1 | tail -5
fi

# --- F16 GGUF conversion ---------------------------------------------------
F16_GGUF="$OUT_DIR/korean-1b-f16.gguf"
echo "Converting to F16 GGUF: $F16_GGUF ..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$HF_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16

echo "F16 GGUF size: $(du -sh "$F16_GGUF" | cut -f1)"

# --- Q4_K_M quantization ---------------------------------------------------
Q4KM_GGUF="$OUT_DIR/korean-1b-q4km.gguf"
if [[ -f "$QUANTIZE_BIN" ]]; then
    echo "Quantizing to Q4_K_M: $Q4KM_GGUF ..."
    "$QUANTIZE_BIN" "$F16_GGUF" "$Q4KM_GGUF" Q4_K_M
    echo "Q4_K_M GGUF size: $(du -sh "$Q4KM_GGUF" | cut -f1)"
else
    echo "[WARN] llama-quantize binary not found. Using F16 GGUF for Ollama."
    echo "       Build: cmake --build $LLAMA_CPP_DIR/build --target llama-quantize"
    cp "$F16_GGUF" "$Q4KM_GGUF"
fi

echo ""
echo "=================================================================="
echo "  GGUF 변환 완료"
echo "  F16 : $F16_GGUF"
echo "  Q4KM: $Q4KM_GGUF"
echo "  다음 단계: bash scripts/deploy_ollama.sh"
echo "=================================================================="
