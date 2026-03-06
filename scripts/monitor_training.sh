#!/usr/bin/env bash
# =============================================================================
# monitor_training.sh — SFT 학습 실시간 모니터링 + 이상 감지
#
# Usage:
#   bash scripts/monitor_training.sh                          # 기본 로그 경로
#   bash scripts/monitor_training.sh /path/to/train.log       # 커스텀 경로
#   bash scripts/monitor_training.sh --check-once             # 1회 검사 후 종료
#
# 감시 항목:
#   🔴 loss = 0.0000 (3 step 연속) → Labels 버그
#   🔴 gnorm > 50.0 → 발산 직전
#   🔴 로그 5분 이상 멈춤 → Hang
#   🟠 loss spike (3× 이동평균) → Bad batch / LR
#   🟠 gnorm > 10.0 → 불안정
#   🟠 디스크 > 80% → 정리 필요
#   🟡 GPU util < 50% → 병목
# =============================================================================
set -euo pipefail

# ---- Configuration ----------------------------------------------------------
LOG_FILE="${1:-checkpoints/korean_1b_sft/train.log}"
CHECK_INTERVAL=30          # 초 단위 폴링 간격
ZERO_LOSS_THRESHOLD=3      # N회 연속 loss=0이면 경고
GNORM_WARN=10.0
GNORM_CRITICAL=50.0
LOSS_SPIKE_FACTOR=3.0      # 이동평균 대비 N배 이상이면 spike
STALL_TIMEOUT=300           # 초 (5분) 로그 멈춤 감지
DISK_WARN_PCT=80
GPU_UTIL_WARN=50
CHECK_ONCE=false

if [[ "${1:-}" == "--check-once" ]]; then
    CHECK_ONCE=true
    LOG_FILE="${2:-checkpoints/korean_1b_sft/train.log}"
fi

# ---- Colors -----------------------------------------------------------------
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---- Helper -----------------------------------------------------------------
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

# ---- Parse last N log lines -------------------------------------------------
parse_metrics() {
    # 로그 형식: [timestamp] [INFO] step  XXXX | loss X.XXXX | lr X.XXe-XX | gnorm X.XXX | ...
    local n="${1:-20}"
    if [[ ! -f "$LOG_FILE" ]]; then
        echo ""
        return
    fi
    tail -n "$n" "$LOG_FILE" | grep "step.*loss.*gnorm" || true
}

extract_field() {
    # $1=line, $2=field name (loss, gnorm, lr)
    echo "$1" | grep -oP "${2}\s+\K[0-9]+\.[0-9e+\-]+" | head -1
}

# ---- Check functions --------------------------------------------------------

check_loss_zero() {
    local lines
    lines=$(parse_metrics "$ZERO_LOSS_THRESHOLD")
    if [[ -z "$lines" ]]; then return; fi

    local zero_count=0
    while IFS= read -r line; do
        local loss
        loss=$(extract_field "$line" "loss")
        if [[ -n "$loss" ]]; then
            # loss < 0.001
            if (( $(echo "$loss < 0.001" | bc -l 2>/dev/null || echo 0) )); then
                ((zero_count++))
            fi
        fi
    done <<< "$lines"

    if [[ $zero_count -ge $ZERO_LOSS_THRESHOLD ]]; then
        alert CRITICAL "Loss가 ${zero_count}회 연속 ~0! Labels 버그 가능성. 즉시 학습 중단!"
        return 1
    fi
    return 0
}

check_loss_spike() {
    local lines
    lines=$(parse_metrics 20)
    if [[ -z "$lines" ]]; then return 0; fi

    local losses=()
    while IFS= read -r line; do
        local loss
        loss=$(extract_field "$line" "loss")
        [[ -n "$loss" ]] && losses+=("$loss")
    done <<< "$lines"

    local count=${#losses[@]}
    if [[ $count -lt 5 ]]; then return 0; fi

    # 마지막 값과 이전 평균 비교
    local last_loss="${losses[$((count-1))]}"
    local sum=0
    for ((i=0; i<count-1; i++)); do
        sum=$(echo "$sum + ${losses[$i]}" | bc -l 2>/dev/null || echo "$sum")
    done
    local avg=$(echo "$sum / ($count - 1)" | bc -l 2>/dev/null || echo "0")

    if [[ "$avg" != "0" ]]; then
        local ratio=$(echo "$last_loss / $avg" | bc -l 2>/dev/null || echo "1")
        if (( $(echo "$ratio > $LOSS_SPIKE_FACTOR" | bc -l 2>/dev/null || echo 0) )); then
            alert WARNING "Loss spike 감지! 현재=${last_loss}, 평균=${avg}, 비율=${ratio}x"
        fi
    fi
    return 0
}

check_gnorm() {
    local lines
    lines=$(parse_metrics 5)
    if [[ -z "$lines" ]]; then return 0; fi

    local last_line
    last_line=$(echo "$lines" | tail -1)
    local gnorm
    gnorm=$(extract_field "$last_line" "gnorm")

    if [[ -z "$gnorm" ]]; then return 0; fi

    if (( $(echo "$gnorm > $GNORM_CRITICAL" | bc -l 2>/dev/null || echo 0) )); then
        alert CRITICAL "GNorm=${gnorm} > ${GNORM_CRITICAL}! 발산 직전. 학습 중단 고려."
    elif (( $(echo "$gnorm > $GNORM_WARN" | bc -l 2>/dev/null || echo 0) )); then
        alert WARNING "GNorm=${gnorm} > ${GNORM_WARN}. 불안정 징후."
    fi
    return 0
}

check_stall() {
    if [[ ! -f "$LOG_FILE" ]]; then
        alert INFO "로그 파일 없음: ${LOG_FILE}"
        return 0
    fi

    local last_modified
    last_modified=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)
    local now
    now=$(date +%s)
    local diff=$((now - last_modified))

    if [[ $diff -gt $STALL_TIMEOUT ]]; then
        alert CRITICAL "로그가 ${diff}초 ($(( diff/60 ))분) 동안 업데이트 없음! Hang 가능성."
    fi
    return 0
}

check_disk() {
    local usage
    usage=$(df /PROJECT 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')
    if [[ -n "$usage" && "$usage" -gt "$DISK_WARN_PCT" ]]; then
        alert WARNING "디스크 사용률 ${usage}% > ${DISK_WARN_PCT}%. 체크포인트 정리 필요."
    fi
    return 0
}

check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then return 0; fi

    local low_util=0
    local total_gpus=0
    while IFS= read -r util; do
        ((total_gpus++))
        if [[ "$util" -lt "$GPU_UTIL_WARN" ]]; then
            ((low_util++))
        fi
    done < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)

    if [[ $total_gpus -gt 0 && $low_util -gt 0 ]]; then
        alert INFO "${low_util}/${total_gpus} GPU utilization < ${GPU_UTIL_WARN}%. 데이터 로딩 병목?"
    fi
    return 0
}

# ---- Status summary ---------------------------------------------------------
print_status() {
    local lines
    lines=$(parse_metrics 1)
    if [[ -n "$lines" ]]; then
        echo -e "${GREEN}최근 로그:${NC} $lines"
    fi

    if command -v nvidia-smi &>/dev/null; then
        echo -e "${CYAN}GPU 메모리:${NC}"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null | head -8
    fi

    local disk
    disk=$(df -h /PROJECT 2>/dev/null | awk 'NR==2 {print "사용: "$3"/"$2" ("$5")"}')
    echo -e "${CYAN}디스크:${NC} ${disk}"
}

# ---- Main loop --------------------------------------------------------------
echo "=================================================================="
echo "  SFT Training Monitor"
echo "  Log file: ${LOG_FILE}"
echo "  Check interval: ${CHECK_INTERVAL}s"
echo "  Press Ctrl+C to stop"
echo "=================================================================="

run_all_checks() {
    check_loss_zero || true
    check_loss_spike || true
    check_gnorm || true
    check_stall || true
    check_disk || true
    check_gpu || true
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
