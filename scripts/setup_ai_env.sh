#!/usr/bin/env bash
# =============================================================================
# setup_ai_env.sh — ai-env 가상환경 생성 + 패키지 설치
#
# Usage:
#   bash scripts/setup_ai_env.sh          # 일반 설치 (CPU/GPU 자동 감지)
#   bash scripts/setup_ai_env.sh --gpu    # NV GPU 서버 (torch는 별도 설치 필요)
#
# Prerequisites:
#   - Python 3.10+ (python3 명령어)
#   - uv (권장) 또는 pip
#   - GPU 서버: NVIDIA 커스텀 torch가 시스템에 설치되어 있어야 함
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
AIENV_DIR="${PROJECT_DIR}/ai-env"
GPU_MODE=false

for arg in "$@"; do
    case "$arg" in
        --gpu) GPU_MODE=true ;;
    esac
done

# ---------- Step 0: Python 탐지 ------------------------------------------------
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "ERROR: python3.10+ not found. Install Python first."
    exit 1
fi

PY_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[0/4] Python: $PYTHON_CMD ($PY_VERSION)"

if [[ "${PY_VERSION%%.*}" -lt 3 ]] || [[ "${PY_VERSION#*.}" -lt 10 ]]; then
    echo "ERROR: Python 3.10+ required, found $PY_VERSION"
    exit 1
fi

# ---------- Step 1: venv 생성 ---------------------------------------------------
if [[ ! -d "$AIENV_DIR" ]]; then
    echo "[1/4] Creating virtual environment at ai-env/..."
    if command -v uv &>/dev/null; then
        uv venv "$AIENV_DIR" --python "$PYTHON_CMD"
    else
        "$PYTHON_CMD" -m venv "$AIENV_DIR"
    fi
else
    echo "[1/4] ai-env already exists, skipping creation."
fi

PY="${AIENV_DIR}/bin/python"
if [[ ! -f "$PY" ]]; then
    echo "ERROR: venv python not found: $PY"
    exit 1
fi

# site-packages 경로 자동 탐지 (python 버전에 독립적)
SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")

# ---------- Step 2: 패키지 설치 -------------------------------------------------
echo "[2/4] Installing packages from requirements.txt..."
if command -v uv &>/dev/null; then
    VIRTUAL_ENV="$AIENV_DIR" uv pip install -r "${PROJECT_DIR}/requirements.txt"
else
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r "${PROJECT_DIR}/requirements.txt"
fi

# ---------- Step 3: NV 커스텀 패키지 심볼릭 링크 (GPU 서버 전용) ------------------
if [[ "$GPU_MODE" == true ]]; then
    echo "[3/4] Linking NV custom packages (GPU mode)..."

    # 시스템 site-packages 자동 탐지
    SYS_SITE=$("$PYTHON_CMD" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    if [[ -z "$SYS_SITE" ]]; then
        # Debian/Ubuntu 폴백
        SYS_SITE="/usr/local/lib/python${PY_VERSION}/dist-packages"
    fi

    for pkg in transformer_engine; do
        if [[ -d "$SYS_SITE/$pkg" ]]; then
            ln -sf "$SYS_SITE/$pkg" "$SITE/$pkg"
            # dist-info 도 링크
            for dist_info in "$SYS_SITE"/${pkg}-*.dist-info; do
                if [[ -d "$dist_info" ]]; then
                    ln -sf "$dist_info" "$SITE/$(basename "$dist_info")"
                fi
            done
            echo "  Linked: $pkg"
        else
            echo "  WARNING: $pkg not found in $SYS_SITE (skip)"
        fi
    done

    # torch가 venv에 없으면 시스템 torch 심볼릭 링크 안내
    if ! "$PY" -c "import torch" 2>/dev/null; then
        echo ""
        echo "  ⚠ torch가 ai-env에 없습니다."
        echo "  NV 커스텀 torch는 수동 설치가 필요합니다:"
        echo "    - 컨테이너 환경: include-system-site-packages=true 설정"
        echo "    - 또는: VIRTUAL_ENV=$AIENV_DIR uv pip install torch --index-url <NV_WHEEL_URL>"
        echo ""
    fi
else
    echo "[3/4] Skipping NV custom packages (non-GPU mode). Use --gpu on NVIDIA servers."
fi

# ---------- Step 4: 검증 --------------------------------------------------------
echo "[4/4] Verifying installation..."
"$PY" -c "
import sys
ok, missing = [], []

# Core packages (requirements.txt)
for pkg in ['transformers','tokenizers','datasets','accelerate','peft','trl',
            'sentencepiece','wandb','tensorboard','yaml','numpy','tqdm']:
    mod = 'PyYAML' if pkg == 'yaml' else pkg
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', 'ok')
        ok.append(f'{pkg} {v}')
    except ImportError:
        missing.append(pkg)

# GPU-only packages (optional)
for pkg in ['torch', 'flash_attn', 'transformer_engine']:
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', 'ok')
        extra = ''
        if pkg == 'torch':
            import torch
            extra = f' CUDA:{torch.cuda.is_available()}'
        ok.append(f'{pkg} {v}{extra}')
    except ImportError:
        ok.append(f'{pkg}: not installed (GPU-only)')

print('Installed:')
for p in ok: print(f'  {p}')
if missing:
    print(f'\nMISSING: {missing}')
    sys.exit(1)
else:
    print('\nAll required packages OK.')
"
