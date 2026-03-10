#!/usr/bin/env bash
# =============================================================================
# check_env.sh — ai-env 환경 검증
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${SCRIPT_DIR}/../ai-env/bin/python"

if [[ ! -f "$PY" ]]; then
    echo "ERROR: ai-env not found. Run: bash scripts/setup_ai_env.sh"
    exit 1
fi

$PY -c "
import torch; print(f'torch {torch.__version__} CUDA:{torch.cuda.is_available()} GPUs:{torch.cuda.device_count()}')
try: import transformer_engine as te; print(f'TE {te.__version__}')
except: print('TE: not installed (GPU-only)')
for pkg in ['trl','tensorboard','peft','accelerate','wandb','datasets','transformers']:
    try: m=__import__(pkg); print(f'{pkg} {getattr(m,\"__version__\",\"ok\")}')
    except: print(f'{pkg}: MISSING')
"
