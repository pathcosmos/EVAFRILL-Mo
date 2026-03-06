#!/usr/bin/env bash
# =============================================================================
# launch_3b_orpo.sh — 8-GPU ORPO fine-tuning launcher for Korean 3B LLM
#
# Usage:
#   bash scripts/launch_3b_orpo.sh                      # 기본 실행
#   bash scripts/launch_3b_orpo.sh --max_steps 200      # 빠른 테스트
#   RUN_NAME=my_orpo bash scripts/launch_3b_orpo.sh     # 이름 지정
#
# 기반 모델 : outputs/hf_korean_3b_sft  (SFT 완료 모델)
# 데이터    : data/preference/combined_preference.jsonl
# 출력      : outputs/korean_3b_orpo_v1/
# 로그      : checkpoints/korean_3b_orpo_v1/train.log
#
# 체크포인트 크기 예상:
#   model weights:    ~6GB (bf16)
#   optimizer states: ~24GB
#   총 ~30GB/개 × max 5개 = 150GB
# =============================================================================
set -euo pipefail

# ---- Configurable defaults --------------------------------------------------
RUN_NAME="${RUN_NAME:-korean_3b_orpo_v1}"
BASE_MODEL="${BASE_MODEL:-outputs/hf_korean_3b_sft}"
DATA_PATH="${DATA_PATH:-data/preference/combined_preference.jsonl}"
OUTPUT_DIR="outputs/${RUN_NAME}"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="${CKPT_DIR}/train.log"
NPROC=8
MASTER_PORT="${MASTER_PORT:-29502}"

# ORPO 하이퍼파라미터
BATCH_SIZE=2
GRAD_ACCUM=4
LR=5e-7
BETA=0.1
EPOCHS=1
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
WARMUP_RATIO=0.05
SEED=42

EXTRA_ARGS="$@"

# ---- B200 / NVSwitch single-node NCCL tuning --------------------------------
# (launch_3b_pretrain.sh와 동일한 NCCL 설정 유지)
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
# ORPO forward-backward 패스는 pretrain보다 메모리 변동이 크므로 버퍼 128MB 유지
export NCCL_BUFFSIZE=134217728
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# OOM 방지: 메모리 단편화 완화 (ORPO는 chosen/rejected 동시 forward → 메모리 민감)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# P2P NVLink 직접 통신 활성화
export NCCL_P2P_LEVEL=NVL
# Ring + Tree 병행 (3B gradient 크기 기준)
export NCCL_ALGO=Ring,Tree

export PYTHONWARNINGS="ignore::UserWarning:torch.library"

cd "$(dirname "$0")/.."

# ---- Pre-flight checks ------------------------------------------------------
if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "ERROR: 기반 모델 디렉토리 없음: ${BASE_MODEL}"
    echo "       SFT 완료 후 HF 포맷으로 변환했는지 확인하세요."
    echo "       예: python scripts/convert_to_hf.py --checkpoint <sft_ckpt> --output ${BASE_MODEL}"
    exit 1
fi

if [[ ! -f "${DATA_PATH}" ]]; then
    echo "ERROR: 학습 데이터 없음: ${DATA_PATH}"
    echo "       먼저 데이터 통합 스크립트를 실행하세요:"
    echo "       python data/prepare_preference_combined.py"
    exit 1
fi

if [[ ! -f "train/orpo.py" ]]; then
    echo "ERROR: train/orpo.py 없음"
    exit 1
fi

# GPU 메모리 체크
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [[ "$GPU_MEM" -gt 0 && "$GPU_MEM" -lt 40000 ]]; then
    echo "WARNING: GPU 메모리 ${GPU_MEM}MB < 40GB. ORPO 3B 학습에 부족할 수 있음."
fi

# 중복 프로세스 방지
EXISTING_PID=$(pgrep -f "orpo.py.*${RUN_NAME}" 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING_PID" ]]; then
    echo "ERROR: 이미 ORPO 프로세스 실행 중 (PID: ${EXISTING_PID})"
    echo "       kill ${EXISTING_PID} 로 먼저 종료하세요."
    exit 1
fi

# 디스크 여유 확인 (최소 200GB)
AVAIL_KB=$(df /PROJECT 2>/dev/null | awk 'NR==2{print $4}' || echo "0")
if [[ -n "$AVAIL_KB" && "$AVAIL_KB" -gt 0 && "$AVAIL_KB" -lt 209715200 ]]; then
    AVAIL_GB=$(echo "scale=1; $AVAIL_KB / 1048576" | bc 2>/dev/null || echo "?")
    echo "WARNING: /PROJECT 여유 ${AVAIL_GB}GB < 200GB. 체크포인트 저장 공간 부족 가능."
fi

mkdir -p "${CKPT_DIR}" "${OUTPUT_DIR}"

# ---- 데이터 레코드 수 확인 --------------------------------------------------
DATA_LINES=$(wc -l < "${DATA_PATH}" 2>/dev/null || echo "?")
echo "  학습 데이터 레코드 수: ${DATA_LINES}"

# ---- 유효 배치 크기 계산 ----------------------------------------------------
EFF_BATCH=$((BATCH_SIZE * NPROC * GRAD_ACCUM))

echo "=================================================================="
echo "  Korean 3B LLM ORPO Fine-Tuning"
echo "  Run name        : ${RUN_NAME}"
echo "  Base model      : ${BASE_MODEL}"
echo "  Data            : ${DATA_PATH}  (${DATA_LINES} records)"
echo "  Output dir      : ${OUTPUT_DIR}"
echo "  CKPT dir        : ${CKPT_DIR}"
echo "  Log file        : ${LOG_FILE}"
echo "  Epochs          : ${EPOCHS}"
echo "  LR              : ${LR}"
echo "  Beta (ORPO)     : ${BETA}"
echo "  Batch           : ${BATCH_SIZE} (local) × ${NPROC} GPU × ${GRAD_ACCUM} accum = ${EFF_BATCH}"
echo "  Max length      : ${MAX_LENGTH}"
echo "  Max prompt len  : ${MAX_PROMPT_LENGTH}"
echo "  Started         : $(date)"
echo "=================================================================="

torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    train/orpo.py \
    --model_path "${BASE_MODEL}" \
    --custom_data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --beta ${BETA} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_length ${MAX_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}" \
         | grep -v "UserWarning" \
         | grep -v "Warning only once" \
         | grep -v "Overriding a previously" \
         | grep -v "dispatch key:" \
         | grep -v "previous kernel:" \
         | grep -v "new kernel:" \
         | grep -v "operator: flash_attn" \
         | grep -v "registered at /usr/local" \
         | grep -v "self.m.impl"

EXIT_CODE=$?
echo "=================================================================="
echo "  Done : $(date)"
echo "  Exit code: ${EXIT_CODE}"
if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo "  모델 저장 위치: ${OUTPUT_DIR}"
fi
echo "=================================================================="
exit $EXIT_CODE
