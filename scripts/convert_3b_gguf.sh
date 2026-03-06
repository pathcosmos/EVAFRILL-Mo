#!/usr/bin/env bash
# =============================================================================
# convert_3b_gguf.sh — 3B 모델 HuggingFace → GGUF 변환 + 다중 양자화
#
# Usage:
#   bash scripts/convert_3b_gguf.sh [options]
#
# Options:
#   --input_dir  DIR   HF 포맷 모델 디렉토리 (default: outputs/hf_korean_3b_orpo)
#   --out_dir    DIR   GGUF 출력 디렉토리    (default: outputs/gguf)
#   --checkpoint DIR   커스텀 체크포인트 디렉토리 (지정 시 HF 변환 선행 실행)
#   --skip_hf_conv     HF 변환 단계 건너뜀 (이미 HF 포맷 존재 시)
#   --skip_quant       양자화 단계 건너뜀 (F16 GGUF만 생성)
#
# Pipeline:
#   1. [선택] 커스텀 체크포인트 → HF transformers 포맷 (convert_to_hf.py)
#   2. HF → F16 GGUF (llama.cpp/convert_hf_to_gguf.py)
#   3. F16 GGUF → Q4_K_M, Q5_K_M, Q8_0 양자화 (llama-quantize)
#
# Outputs:
#   outputs/gguf/frankenstallm-3b-f16.gguf
#   outputs/gguf/frankenstallm-3b-Q4_K_M.gguf   — 권장 (Ollama용)
#   outputs/gguf/frankenstallm-3b-Q5_K_M.gguf
#   outputs/gguf/frankenstallm-3b-Q8_0.gguf
#
# 전제 조건:
#   - python scripts/convert_to_hf.py 로 HF 변환 완료 (또는 --checkpoint 옵션)
#   - git, cmake, make 설치
#   - pip install safetensors
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------
INPUT_DIR="outputs/hf_korean_3b_orpo"
OUT_DIR="outputs/gguf"
CHECKPOINT_DIR=""
SKIP_HF_CONV=false
SKIP_QUANT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)   INPUT_DIR="$2";      shift 2 ;;
        --out_dir)     OUT_DIR="$2";        shift 2 ;;
        --checkpoint)  CHECKPOINT_DIR="$2"; shift 2 ;;
        --skip_hf_conv) SKIP_HF_CONV=true; shift ;;
        --skip_quant)   SKIP_QUANT=true;   shift ;;
        -h|--help)
            grep '^#' "$0" | head -40 | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "ERROR: 알 수 없는 옵션: $1"
            echo "Usage: bash scripts/convert_3b_gguf.sh [--input_dir DIR] [--out_dir DIR] [--checkpoint DIR] [--skip_hf_conv] [--skip_quant]"
            exit 1 ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_DIR/outputs/llama.cpp}"
MODEL_NAME="frankenstallm-3b"

cd "$PROJECT_DIR"

echo "=================================================================="
echo "  3B 모델 GGUF 변환 파이프라인"
echo "  입력 HF 디렉토리 : $INPUT_DIR"
echo "  GGUF 출력 디렉토리: $OUT_DIR"
echo "  llama.cpp 경로    : $LLAMA_CPP_DIR"
echo "=================================================================="
echo ""

# ---------------------------------------------------------------------------
# Step 0: llama.cpp 존재 여부 확인 / 클론
# ---------------------------------------------------------------------------
if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    echo "[SETUP] llama.cpp 디렉토리가 없습니다."
    echo "        다음 명령으로 설치하세요:"
    echo ""
    echo "        git clone --depth 1 https://github.com/ggerganov/llama.cpp $LLAMA_CPP_DIR"
    echo ""
    echo "        또는 LLAMA_CPP_DIR 환경변수로 기존 경로를 지정하세요:"
    echo "        LLAMA_CPP_DIR=/path/to/llama.cpp bash scripts/convert_3b_gguf.sh"
    echo ""
    read -r -p "지금 자동 클론하시겠습니까? [y/N] " _yn
    if [[ "${_yn:-N}" =~ ^[Yy]$ ]]; then
        echo "Cloning llama.cpp ..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
    else
        echo "중단합니다. llama.cpp를 설치한 뒤 다시 실행하세요."
        exit 1
    fi
fi

# llama.cpp Python 의존성
echo "[SETUP] llama.cpp Python 의존성 설치 중 ..."
pip install -r "$LLAMA_CPP_DIR/requirements.txt" --break-system-packages -q

# ---------------------------------------------------------------------------
# Step 1: 커스텀 체크포인트 → HF 포맷 변환 (선택)
# ---------------------------------------------------------------------------
if [[ -n "$CHECKPOINT_DIR" && "$SKIP_HF_CONV" == "false" ]]; then
    echo ""
    echo "[STEP 1] 커스텀 체크포인트 → HF 포맷 변환"
    echo "  체크포인트: $CHECKPOINT_DIR"
    echo "  출력      : $INPUT_DIR"
    echo ""

    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        echo "ERROR: 체크포인트 디렉토리를 찾을 수 없습니다: $CHECKPOINT_DIR"
        exit 1
    fi

    python "$PROJECT_DIR/scripts/convert_to_hf.py" \
        --checkpoint "$CHECKPOINT_DIR" \
        --output "$INPUT_DIR" \
        --tokenizer "tokenizer/korean_sp/tokenizer.json"

    echo "  [OK] HF 변환 완료 → $INPUT_DIR"
elif [[ "$SKIP_HF_CONV" == "true" ]]; then
    echo "[STEP 1] HF 변환 건너뜀 (--skip_hf_conv)"
else
    echo "[STEP 1] 체크포인트 미지정 — HF 디렉토리를 직접 사용합니다."
fi

# HF 디렉토리 최종 검증
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: HF 모델 디렉토리를 찾을 수 없습니다: $INPUT_DIR"
    echo "  --checkpoint 옵션으로 체크포인트를 지정하거나,"
    echo "  python scripts/convert_to_hf.py 를 먼저 실행하세요."
    exit 1
fi

if [[ ! -f "$INPUT_DIR/config.json" ]]; then
    echo "ERROR: config.json 이 없습니다: $INPUT_DIR/config.json"
    exit 1
fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Step 2: llama.cpp 빌드 (llama-quantize 바이너리)
# ---------------------------------------------------------------------------
QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"

if [[ ! -f "$QUANTIZE_BIN" ]]; then
    echo ""
    echo "[STEP 2] llama.cpp 빌드 중 (llama-quantize) ..."
    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        2>&1 | tail -10
    cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j "$(nproc)" \
        2>&1 | tail -10
    echo "  [OK] 빌드 완료: $QUANTIZE_BIN"
else
    echo "[STEP 2] llama-quantize 바이너리 이미 존재 — 빌드 건너뜀"
fi

# ---------------------------------------------------------------------------
# Step 3: HF → F16 GGUF 변환
# ---------------------------------------------------------------------------
F16_GGUF="$OUT_DIR/${MODEL_NAME}-f16.gguf"

echo ""
echo "[STEP 3] HF → F16 GGUF 변환"
echo "  입력: $INPUT_DIR"
echo "  출력: $F16_GGUF"
echo ""

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$INPUT_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16

echo "  [OK] F16 GGUF 크기: $(du -sh "$F16_GGUF" | cut -f1)  ($F16_GGUF)"

# ---------------------------------------------------------------------------
# Step 4: 다중 양자화 (Q4_K_M, Q5_K_M, Q8_0)
# ---------------------------------------------------------------------------
if [[ "$SKIP_QUANT" == "true" ]]; then
    echo ""
    echo "[STEP 4] 양자화 건너뜀 (--skip_quant)"
else
    echo ""
    echo "[STEP 4] 다중 양자화 시작 ..."

    if [[ ! -f "$QUANTIZE_BIN" ]]; then
        echo "[WARN] llama-quantize 바이너리를 찾을 수 없습니다: $QUANTIZE_BIN"
        echo "       양자화를 건너뜁니다. F16 GGUF만 생성되었습니다."
        echo "       수동 빌드: cmake --build $LLAMA_CPP_DIR/build --target llama-quantize"
    else
        # Q4_K_M — 가장 작은 크기, 품질/속도 균형 (Ollama 기본 권장)
        Q4KM_GGUF="$OUT_DIR/${MODEL_NAME}-Q4_K_M.gguf"
        echo "  → Q4_K_M 양자화: $Q4KM_GGUF ..."
        "$QUANTIZE_BIN" "$F16_GGUF" "$Q4KM_GGUF" Q4_K_M
        echo "     크기: $(du -sh "$Q4KM_GGUF" | cut -f1)"

        # Q5_K_M — 중간 크기, 더 높은 품질
        Q5KM_GGUF="$OUT_DIR/${MODEL_NAME}-Q5_K_M.gguf"
        echo "  → Q5_K_M 양자화: $Q5KM_GGUF ..."
        "$QUANTIZE_BIN" "$F16_GGUF" "$Q5KM_GGUF" Q5_K_M
        echo "     크기: $(du -sh "$Q5KM_GGUF" | cut -f1)"

        # Q8_0 — 가장 높은 품질 (F16 근사)
        Q8_GGUF="$OUT_DIR/${MODEL_NAME}-Q8_0.gguf"
        echo "  → Q8_0 양자화: $Q8_GGUF ..."
        "$QUANTIZE_BIN" "$F16_GGUF" "$Q8_GGUF" Q8_0
        echo "     크기: $(du -sh "$Q8_GGUF" | cut -f1)"

        echo ""
        echo "  [OK] 모든 양자화 완료"
    fi
fi

# ---------------------------------------------------------------------------
# 완료 요약
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  3B GGUF 변환 완료"
echo ""
echo "  출력 파일 목록:"
ls -lh "$OUT_DIR/${MODEL_NAME}"*.gguf 2>/dev/null | awk '{print "    " $5 "  " $9}' || \
    echo "    (파일 목록 확인: ls -lh $OUT_DIR/)"
echo ""
echo "  다음 단계:"
echo "    bash scripts/deploy_3b_ollama.sh"
echo "    bash scripts/quality_gate.sh deploy"
echo "=================================================================="
