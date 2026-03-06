#!/usr/bin/env bash
# =============================================================================
# monitor_3b.sh — 3B 학습 실시간 모니터링 + 이상 감지 + 자동 체크포인트 정리
#
# Usage:
#   bash scripts/monitor_3b.sh                          # 기본 감시
#   bash scripts/monitor_3b.sh --check-once             # 1회 검사
#   bash scripts/monitor_3b.sh --auto-cleanup           # 자동 오래된 체크포인트 삭제
#
# 3B 특화 사항:
#   - 체크포인트 27GB/개 → 디스크 감시 강화
#   - NCCL hang 감지 + 자동 재시작 옵션
#   - 예상 완료 시간 실시간 계산
#   - 프로세스 중복 실행 방지
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
RUN_NAME="${RUN_NAME:-korean_3b_fp8_run1}"
LOG_FILE="${1:-checkpoints/${RUN_NAME}/train.log}"
CKPT_DIR="checkpoints/${RUN_NAME}"
CHECK_INTERVAL=60          # 3B는 step 간격 더 김 → 60초
ZERO_LOSS_THRESHOLD=3
GNORM_WARN=10.0
GNORM_CRITICAL=50.0
LOSS_SPIKE_FACTOR=3.0
STALL_TIMEOUT=600           # 10분 (3B는 step 더 오래 걸림)
DISK_WARN_PCT=85
DISK_CRITICAL_PCT=92
GPU_UTIL_WARN=50
MAX_CHECKPOINTS=15          # 최대 보관 체크포인트 수 (15 × 27GB = 405GB)
CHECK_ONCE=false
AUTO_CLEANUP=false
AUTO_RESTART=false

# Parse args
for arg in "$@"; do
    case "$arg" in
        --check-once)   CHECK_ONCE=true ;;
        --auto-cleanup) AUTO_CLEANUP=true ;;
        --auto-restart) AUTO_RESTART=true ;;
    esac
done
# Fix LOG_FILE if first arg was a flag
if [[ "$LOG_FILE" == --* ]]; then
    LOG_FILE="checkpoints/${RUN_NAME}/train.log"
fi

# ---- Colors -----------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; NC='\033[0m'

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

alert() {
    local level="$1" msg="$2"
    case "$level" in
        CRITICAL) echo -e "${RED}🔴 [$(timestamp)] [CRITICAL] ${msg}${NC}" ;;
        WARNING)  echo -e "${YELLOW}🟠 [$(timestamp)] [WARNING]  ${msg}${NC}" ;;
        INFO)     echo -e "${CYAN}🟡 [$(timestamp)] [INFO]     ${msg}${NC}" ;;
        OK)       echo -e "${GREEN}✅ [$(timestamp)] [OK]       ${msg}${NC}" ;;
    esac
}

# ---- Parse metrics ----------------------------------------------------------
parse_metrics() {
    local n="${1:-20}"
    [[ -f "$LOG_FILE" ]] || return
    tail -n "$n" "$LOG_FILE" | grep "step.*loss.*gnorm" || true
}

extract_field() {
    echo "$1" | grep -oP "${2}\s+\K[0-9]+\.[0-9e+\-]+" | head -1
}

extract_step() {
    echo "$1" | grep -oP "step\s+\K[0-9]+" | head -1
}

# ---- Check: Loss = 0 -------------------------------------------------------
check_loss_zero() {
    local lines
    lines=$(parse_metrics "$ZERO_LOSS_THRESHOLD")
    [[ -z "$lines" ]] && return 0
    local zero_count=0
    while IFS= read -r line; do
        local loss=$(extract_field "$line" "loss")
        if [[ -n "$loss" ]] && (( $(echo "$loss < 0.001" | bc -l 2>/dev/null || echo 0) )); then
            ((zero_count++))
        fi
    done <<< "$lines"
    if [[ $zero_count -ge $ZERO_LOSS_THRESHOLD ]]; then
        alert CRITICAL "Loss가 ${zero_count}회 연속 ~0! Labels 버그. 즉시 중단!"
        return 1
    fi
}

# ---- Check: Loss spike -----------------------------------------------------
check_loss_spike() {
    local lines=$(parse_metrics 20)
    [[ -z "$lines" ]] && return 0
    local losses=()
    while IFS= read -r line; do
        local loss=$(extract_field "$line" "loss")
        [[ -n "$loss" ]] && losses+=("$loss")
    done <<< "$lines"
    local count=${#losses[@]}
    [[ $count -lt 5 ]] && return 0
    local last="${losses[$((count-1))]}"
    local sum=0
    for ((i=0; i<count-1; i++)); do
        sum=$(echo "$sum + ${losses[$i]}" | bc -l 2>/dev/null || echo "$sum")
    done
    local avg=$(echo "$sum / ($count - 1)" | bc -l 2>/dev/null || echo "0")
    if [[ "$avg" != "0" ]]; then
        local ratio=$(echo "$last / $avg" | bc -l 2>/dev/null || echo "1")
        if (( $(echo "$ratio > $LOSS_SPIKE_FACTOR" | bc -l 2>/dev/null || echo 0) )); then
            alert WARNING "Loss spike! 현재=${last}, 평균=${avg}, 비율=${ratio}x"
        fi
    fi
}

# ---- Check: Gradient norm ---------------------------------------------------
check_gnorm() {
    local lines=$(parse_metrics 5)
    [[ -z "$lines" ]] && return 0
    local gnorm=$(extract_field "$(echo "$lines" | tail -1)" "gnorm")
    [[ -z "$gnorm" ]] && return 0
    if (( $(echo "$gnorm > $GNORM_CRITICAL" | bc -l 2>/dev/null || echo 0) )); then
        alert CRITICAL "GNorm=${gnorm} > ${GNORM_CRITICAL}! 발산 직전."
    elif (( $(echo "$gnorm > $GNORM_WARN" | bc -l 2>/dev/null || echo 0) )); then
        alert WARNING "GNorm=${gnorm} 불안정."
    fi
}

# ---- Check: Stall / NCCL hang ----------------------------------------------
check_stall() {
    [[ ! -f "$LOG_FILE" ]] && return 0
    local last_mod=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)
    local now=$(date +%s)
    local diff=$((now - last_mod))
    if [[ $diff -gt $STALL_TIMEOUT ]]; then
        alert CRITICAL "로그 ${diff}초 ($(( diff/60 ))분) 멈춤! NCCL hang 가능성."
        # NCCL hang 자동 재시작
        if $AUTO_RESTART; then
            alert WARNING "자동 재시작 시도..."
            local pid=$(pgrep -f "pretrain.py.*korean_3b" | head -1 || true)
            if [[ -n "$pid" ]]; then
                kill -9 "$pid" 2>/dev/null || true
                sleep 5
                alert INFO "이전 프로세스 종료. launch_3b_pretrain.sh 재실행 필요."
            fi
        fi
    fi
}

# ---- Check: Disk (3B 강화) --------------------------------------------------
check_disk() {
    local usage=$(df /PROJECT 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')
    if [[ -n "$usage" && "$usage" -gt "$DISK_CRITICAL_PCT" ]]; then
        alert CRITICAL "디스크 ${usage}% > ${DISK_CRITICAL_PCT}%! 즉시 정리 필요!"
        $AUTO_CLEANUP && cleanup_old_checkpoints
    elif [[ -n "$usage" && "$usage" -gt "$DISK_WARN_PCT" ]]; then
        alert WARNING "디스크 ${usage}% > ${DISK_WARN_PCT}%. 체크포인트 정리 권장."
    fi
}

# ---- Check: GPU utilization -------------------------------------------------
check_gpu() {
    command -v nvidia-smi &>/dev/null || return 0
    local low=0 total=0
    while IFS= read -r util; do
        ((total++))
        [[ "$util" -lt "$GPU_UTIL_WARN" ]] && ((low++))
    done < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    [[ $total -gt 0 && $low -gt 0 ]] && alert INFO "${low}/${total} GPU util < ${GPU_UTIL_WARN}%"
}

# ---- Check: 체크포인트 무결성 -----------------------------------------------
check_checkpoint_integrity() {
    local latest=$(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    [[ -z "$latest" ]] && return 0
    # 최소 파일 존재 확인
    if [[ ! -f "${latest}/model.pt" ]] && [[ ! -f "${latest}/model.safetensors" ]]; then
        alert WARNING "최근 체크포인트에 모델 파일 없음: ${latest}"
    fi
    # 크기 확인 (3B model.pt는 최소 2GB)
    local size=$(du -sb "${latest}" 2>/dev/null | awk '{print $1}')
    if [[ -n "$size" && "$size" -lt 2000000000 ]]; then
        alert WARNING "체크포인트 크기 비정상 (${size} bytes < 2GB): ${latest}"
    fi
}

# ---- Cleanup: 오래된 체크포인트 자동 삭제 ------------------------------------
cleanup_old_checkpoints() {
    local ckpts=($(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | sort -V))
    local count=${#ckpts[@]}
    if [[ $count -le $MAX_CHECKPOINTS ]]; then
        alert OK "체크포인트 ${count}개 ≤ ${MAX_CHECKPOINTS}. 정리 불필요."
        return
    fi
    # 이정표 체크포인트 보존 (매 10K step)
    local deletable=()
    local preserved=()
    for ckpt in "${ckpts[@]}"; do
        local step_num=$(basename "$ckpt" | grep -oP '\d+' || echo "0")
        if (( step_num % 10000 == 0 && step_num > 0 )); then
            preserved+=("$ckpt")
        else
            deletable+=("$ckpt")
        fi
    done
    # 최근 MAX_CHECKPOINTS개는 무조건 보존
    local n_deletable=${#deletable[@]}
    local total_keep=$(( ${#preserved[@]} + MAX_CHECKPOINTS ))
    local to_delete=$(( count - total_keep ))
    [[ $to_delete -le 0 ]] && { alert OK "정리 불필요 (이정표 ${#preserved[@]}개 + 최근 ${MAX_CHECKPOINTS}개 보존)."; return; }
    alert INFO "${count}개 체크포인트 → ${to_delete}개 삭제 (이정표 ${#preserved[@]}개 영구 보존)"
    local deleted=0
    for ckpt in "${deletable[@]}"; do
        [[ $deleted -ge $to_delete ]] && break
        local ckpt_size=$(du -sh "$ckpt" 2>/dev/null | awk '{print $1}')
        echo "  삭제: $ckpt (${ckpt_size})"
        rm -rf "$ckpt"
        ((deleted++))
    done
    alert OK "체크포인트 정리 완료. (${deleted}개 삭제)"
}

# ---- ETA 계산 ---------------------------------------------------------------
estimate_eta() {
    [[ ! -f "$LOG_FILE" ]] && return
    # 최근 step 번호 + 시간
    local lines=$(parse_metrics 50)
    [[ -z "$lines" ]] && return
    local last_line=$(echo "$lines" | tail -1)
    local first_line=$(echo "$lines" | head -1)
    local cur_step=$(extract_step "$last_line")
    local max_steps=$(grep -oP "max_steps.*?(\d+)" "${CKPT_DIR}/train.log" 2>/dev/null | head -1 | grep -oP '\d+$' || echo "57000")

    [[ -z "$cur_step" || "$cur_step" == "0" ]] && return

    # step/sec from log timestamps (approximate)
    local remaining=$((max_steps - cur_step))
    if [[ $remaining -le 0 ]]; then
        echo -e "${MAGENTA}📊 진행: ${cur_step}/${max_steps} (완료!)${NC}"
        return
    fi

    # 파일 수정 시간 기반 rough ETA
    local first_time=$(head -20 "$LOG_FILE" 2>/dev/null | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1 || true)
    if [[ -n "$first_time" ]]; then
        local start_epoch=$(date -d "$first_time" +%s 2>/dev/null || echo 0)
        local now=$(date +%s)
        if [[ $start_epoch -gt 0 && $cur_step -gt 0 ]]; then
            local elapsed=$((now - start_epoch))
            local sec_per_step=$(echo "$elapsed / $cur_step" | bc -l 2>/dev/null || echo "0")
            local eta_sec=$(echo "$remaining * $sec_per_step" | bc 2>/dev/null | cut -d. -f1 || echo "0")
            local eta_hours=$(echo "$eta_sec / 3600" | bc 2>/dev/null || echo "?")
            local pct=$(echo "scale=1; $cur_step * 100 / $max_steps" | bc 2>/dev/null || echo "?")
            echo -e "${MAGENTA}📊 진행: ${cur_step}/${max_steps} (${pct}%) | 남은 시간: ~${eta_hours}h | ${sec_per_step}s/step${NC}"
        fi
    else
        echo -e "${MAGENTA}📊 진행: ${cur_step}/${max_steps}${NC}"
    fi
}

# ---- Status summary ---------------------------------------------------------
print_status() {
    local lines=$(parse_metrics 1)
    [[ -n "$lines" ]] && echo -e "${GREEN}최근:${NC} $lines"
    estimate_eta
    if command -v nvidia-smi &>/dev/null; then
        echo -e "${CYAN}GPU:${NC}"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
            --format=csv,noheader 2>/dev/null | head -8
    fi
    local ckpt_count=$(ls -d "${CKPT_DIR}"/checkpoint-* 2>/dev/null | wc -l)
    local ckpt_size=$(du -sh "${CKPT_DIR}" 2>/dev/null | awk '{print $1}')
    echo -e "${CYAN}체크포인트:${NC} ${ckpt_count}개 (${ckpt_size})"
    local disk=$(df -h /PROJECT 2>/dev/null | awk 'NR==2 {print $3"/"$2" ("$5")"}')
    echo -e "${CYAN}디스크:${NC} ${disk}"
}

# ---- Main -------------------------------------------------------------------
echo "=================================================================="
echo "  3B Training Monitor"
echo "  Run: ${RUN_NAME}"
echo "  Log: ${LOG_FILE}"
echo "  Interval: ${CHECK_INTERVAL}s"
echo "  Auto-cleanup: ${AUTO_CLEANUP} | Auto-restart: ${AUTO_RESTART}"
echo "  Ctrl+C to stop"
echo "=================================================================="

run_all_checks() {
    check_loss_zero || true
    check_loss_spike || true
    check_gnorm || true
    check_stall || true
    check_disk || true
    check_gpu || true
    check_checkpoint_integrity || true
    echo "---"
    print_status
    echo ""
}

if $CHECK_ONCE; then
    run_all_checks
    exit 0
fi

while true; do
    run_all_checks
    sleep "$CHECK_INTERVAL"
done
