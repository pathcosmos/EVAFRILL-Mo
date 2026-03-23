#!/usr/bin/env bash
# =============================================================================
# launch_3b_pretrain.sh — 8-GPU FP8 pretraining launcher for Korean 3B LLM
#
# Features:
#   - SIGHUP 방어: SSH 끊김 시 자동으로 nohup+setsid로 세션 보호
#   - Graceful shutdown: SIGTERM 시 Python 시그널 핸들러가 비상 체크포인트 저장
#   - 자동 resume: 최신 체크포인트에서 자동 재개
#   - PID 파일: 프로세스 모니터링 및 제어용
#   - grep 파이프라인 exit code 보호 (|| true)
#
# Usage:
#   bash scripts/launch_3b_pretrain.sh                          # full run (60B tokens)
#   bash scripts/launch_3b_pretrain.sh --max_steps 500          # quick test
#   bash scripts/launch_3b_pretrain.sh --resume checkpoints/korean_3b_fp8_run1/checkpoint-0010000
#   MAX_STEPS=95000 bash scripts/launch_3b_pretrain.sh          # 100B tokens
#
# 모니터링:
#   tail -f checkpoints/korean_3b_fp8_run1/train.log
#   cat checkpoints/korean_3b_fp8_run1/train.pid
#
# 중지 (비상 체크포인트 자동 저장):
#   kill $(cat checkpoints/korean_3b_fp8_run1/train.pid)
#
# 강제 종료 (체크포인트 저장 없음):
#   kill -9 $(cat checkpoints/korean_3b_fp8_run1/train.pid)
# =============================================================================

# -u: 미정의 변수 에러
# NOTE: -e, -o pipefail 의도적 제거
#   이전 문제: grep 파이프라인에서 모든 라인이 필터링되면 exit code 1 반환
#   → pipefail이 이를 스크립트 실패로 전파 → 학습 중단
#   해결: set -e/pipefail 제거 + grep 체인에 || true 추가
set -u

# ---- Configurable defaults --------------------------------------------------
RUN_NAME="${RUN_NAME:-korean_3b_fp8_run1}"
CONFIG="${CONFIG:-configs/korean_3b_fp8.yaml}"
TRAIN_DATA="${TRAIN_DATA:-data/3b_train.bin}"
VAL_DATA="${VAL_DATA:-data/3b_val.bin}"
CKPT_DIR="checkpoints/${RUN_NAME}"
LOG_FILE="${CKPT_DIR}/train.log"
NPROC=8
MASTER_PORT="${MASTER_PORT:-29501}"

MAX_STEPS="${MAX_STEPS:-57000}"
BATCH_SIZE=5
GRAD_ACCUM=8
WARMUP_STEPS=2000
SEED=42

# ---- B200 / NVSwitch single-node NCCL tuning (3B optimized, v2) ----------
export NCCL_IB_DISABLE=1
export NCCL_ALGO=NVLS,Ring              # NVSwitch hardware reduction first (was Ring,Tree)
export NCCL_PROTO=Simple
export NCCL_NVLS_ENABLE=1               # NVLink SHARP — hardware-accelerated all-reduce
export NCCL_MIN_NCHANNELS=32            # raise minimum for NVSwitch headroom (was 16)
export NCCL_MAX_NCHANNELS=32
export NCCL_BUFFSIZE=268435456          # 256MB (was 128MB) — reduces bucket pipeline stalls
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Triton/Inductor cache on executable filesystem (not /tmp which is noexec)
export TRITON_CUDACRT_PATH=/usr/local/cuda/include
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

cd "$(dirname "$0")/.."

mkdir -p "${CKPT_DIR}"

# ---- Session protection (SIGHUP 방어) ---------------------------------------
# tmux/screen 없이 실행 시, 자동으로 nohup + setsid로 래핑하여
# SSH 끊김(SIGHUP)으로부터 학습 프로세스를 보호합니다.
#
# 작동 원리:
#   1. tmux/screen/이미 보호됨 여부 확인
#   2. 미보호 상태이면 _LAUNCH_PROTECTED=1 설정 후 nohup setsid로 자기 자신을 재실행
#   3. 재실행된 프로세스는 새로운 세션 리더가 되어 터미널과 분리됨
#   4. 원래 셸은 PID와 모니터링 명령을 출력하고 즉시 종료
PID_FILE="${CKPT_DIR}/train.pid"

if [[ -z "${_LAUNCH_PROTECTED:-}" ]] && [[ -z "${TMUX:-}" ]] && [[ -z "${STY:-}" ]]; then
    export _LAUNCH_PROTECTED=1
    NOHUP_LOG="${CKPT_DIR}/launch_$(date +%Y%m%d_%H%M%S).log"

    echo "=================================================================="
    echo "  SIGHUP PROTECTION ACTIVATED"
    echo "  tmux/screen 미감지 → 세션 보호 모드 자동 활성화 (nohup + setsid)"
    echo "  SSH 끊어져도 학습이 계속됩니다."
    echo "=================================================================="
    echo ""

    # 자기 자신을 세션 보호 모드로 재실행
    nohup setsid bash "$0" "$@" > "${NOHUP_LOG}" 2>&1 &
    BG_PID=$!
    echo "${BG_PID}" > "${PID_FILE}"

    echo "  PID         : ${BG_PID}"
    echo "  PID 파일    : ${PID_FILE}"
    echo "  Launch 로그 : ${NOHUP_LOG}"
    echo "  학습 로그   : ${LOG_FILE}"
    echo ""
    echo "  모니터링:"
    echo "    tail -f ${LOG_FILE}"
    echo ""
    echo "  중지 (비상 체크포인트 자동 저장):"
    echo "    kill \$(cat ${PID_FILE})"
    echo ""
    echo "  강제 종료:"
    echo "    kill -9 \$(cat ${PID_FILE})"
    echo "=================================================================="
    exit 0
fi

# ---- Cleanup on exit --------------------------------------------------------
PREWARM_PID=""

cleanup() {
    rm -f "${PID_FILE}" 2>/dev/null || true
    if [[ -n "${PREWARM_PID:-}" ]]; then
        kill "${PREWARM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# PID 파일 기록 (tmux/screen 내에서 실행 시에도 PID 추적 가능)
echo "$$" > "${PID_FILE}"

# ---- Pre-flight checks ------------------------------------------------------
if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG}"
    exit 1
fi

if [[ ! -f "${TRAIN_DATA}" ]]; then
    echo "[ERROR] Training data not found: ${TRAIN_DATA}"
    exit 1
fi

# GPU 메모리 체크 (3B는 최소 80GB/GPU 권장, B200=192GB → OK)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [[ "$GPU_MEM" -gt 0 && "$GPU_MEM" -lt 80000 ]]; then
    echo "[WARN] GPU memory ${GPU_MEM}MB < 80GB. 3B 학습에 부족할 수 있음."
fi

# 중복 프로세스 방지
EXISTING_PID=$(pgrep -f "pretrain.py.*korean_3b" 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING_PID" ]]; then
    echo "[ERROR] 이미 3B pretrain 프로세스 실행 중 (PID: ${EXISTING_PID})"
    echo "        kill ${EXISTING_PID} 로 먼저 종료하세요."
    exit 1
fi

# 디스크 여유 확인 (최소 1TB 필요)
AVAIL_KB=$(df /PROJECT 2>/dev/null | awk 'NR==2{print $4}')
if [[ -n "${AVAIL_KB:-}" ]] && [[ "$AVAIL_KB" -lt 1073741824 ]]; then
    AVAIL_TB=$(echo "scale=1; $AVAIL_KB / 1073741824" | bc 2>/dev/null || echo "?")
    echo "[WARN] /PROJECT 여유 ${AVAIL_TB}TB < 1TB. 체크포인트 저장 공간 부족 가능."
fi

# ---- Resume detection -------------------------------------------------------
RESUME_ARG=""
EXTRA_ARGS="${*:-}"
if [[ ! "${EXTRA_ARGS}" =~ "--resume" ]]; then
    # 가장 최근 체크포인트 자동 감지
    LATEST_CKPT=$(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "$LATEST_CKPT" ]]; then
        echo "[INFO] 자동 resume 감지: ${LATEST_CKPT}"
        RESUME_ARG="--resume ${LATEST_CKPT}"
    fi
fi

# ---- Banner ------------------------------------------------------------------
SESSION_TYPE="direct"
[[ -n "${TMUX:-}" ]] && SESSION_TYPE="tmux"
[[ -n "${STY:-}" ]] && SESSION_TYPE="screen"
[[ -n "${_LAUNCH_PROTECTED:-}" ]] && SESSION_TYPE="protected (nohup+setsid)"

echo "=================================================================="
echo "  Korean 3B LLM Pre-Training (FP8)"
echo "  Run name    : ${RUN_NAME}"
echo "  Config      : ${CONFIG}"
echo "  CKPT dir    : ${CKPT_DIR}"
echo "  Log file    : ${LOG_FILE}"
echo "  Max steps   : ${MAX_STEPS}"
echo "  Batch       : ${BATCH_SIZE} (local) x ${NPROC} GPU x ${GRAD_ACCUM} accum"
echo "  Eff tokens  : $((BATCH_SIZE * NPROC * GRAD_ACCUM * 4096)) tokens/step (~1M)"
echo "  Total tokens: ~$((MAX_STEPS * BATCH_SIZE * NPROC * GRAD_ACCUM * 4096 / 1000000000))B"
echo "  Resume      : ${RESUME_ARG:-none (fresh start)}"
echo "  Session     : ${SESSION_TYPE}"
echo "  PID         : $$ (file: ${PID_FILE})"
echo "  Started     : $(date)"
echo "=================================================================="

export PYTHONWARNINGS="ignore::UserWarning:torch.library"

# ---- Pre-warm OS page cache (NUMA-interleaved, non-blocking) ---------------
if [[ -f "${TRAIN_DATA}" ]]; then
    echo "[INFO] Pre-warming page cache for ${TRAIN_DATA} (NUMA interleaved)..."
    numactl --interleave=all dd if="${TRAIN_DATA}" of=/dev/null bs=16M 2>/dev/null &
    PREWARM_PID=$!
fi

# ---- Launch training ---------------------------------------------------------
# grep 파이프라인 보호:
#   문제: grep -v 가 매칭 라인이 없으면 exit code 1 반환
#   해결: { ... || true; } 래핑으로 파이프라인 exit code를 항상 0으로 보장
#   torchrun의 실제 exit code는 PIPESTATUS[0]으로 별도 캡처
numactl --interleave=all \
torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    train/pretrain.py \
    --config "${CONFIG}" \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --log_file "${LOG_FILE}" \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --warmup_steps ${WARMUP_STEPS} \
    --seed ${SEED} \
    ${RESUME_ARG} \
    ${EXTRA_ARGS} \
    2>&1 | { grep -v "UserWarning" \
           | grep -v "Warning only once" \
           | grep -v "Overriding a previously" \
           | grep -v "dispatch key:" \
           | grep -v "previous kernel:" \
           | grep -v "new kernel:" \
           | grep -v "operator: flash_attn" \
           | grep -v "registered at /usr/local" \
           | grep -v "self.m.impl" \
           || true; }

EXIT_CODE=${PIPESTATUS[0]}

# ---- Exit summary ------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  Finished  : $(date)"
echo "  Exit code : ${EXIT_CODE}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "  Status    : SUCCESS (학습 완료 또는 graceful shutdown)"
elif [[ ${EXIT_CODE} -eq 143 ]]; then
    echo "  Status    : TERMINATED (SIGTERM — 비상 체크포인트 저장됨)"
elif [[ ${EXIT_CODE} -eq 137 ]]; then
    echo "  Status    : KILLED (SIGKILL — 강제 종료, 체크포인트 미저장)"
elif [[ ${EXIT_CODE} -eq 1 ]]; then
    echo "  Status    : ERROR (${LOG_FILE} 확인 필요)"
else
    echo "  Status    : FAILED (exit code ${EXIT_CODE}, ${LOG_FILE} 확인)"
fi
echo "=================================================================="
exit ${EXIT_CODE}
