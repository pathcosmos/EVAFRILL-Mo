#!/usr/bin/env bash
# =============================================================================
# finish_korean_pipeline.sh
# 한국어 LLM 데이터 파이프라인 Step 6~7 재개 스크립트
#
# - 완료된 단계(출력 파일 존재)는 자동으로 건너뜀
# - --from-step N 지정 시 해당 스텝부터 강제 재실행
# - 상세 로그를 파일 + 터미널에 동시 출력
#
# 스텝 번호:
#   61 = Step 6a : c4_ko 토크나이징
#   62 = Step 6b : namuwiki_ko 토크나이징
#   63 = Step 6c : ko_wiki 토크나이징
#   70 = Step 7  : 병합 (korean_train.bin / korean_val.bin)
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# 프로젝트 루트로 이동 (스크립트 위치 기준으로 한 단계 위)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# 인자 파싱
# -----------------------------------------------------------------------------
FROM_STEP=0
LOG_FILE="data/finish_korean_pipeline.log"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --from-step)
            FROM_STEP="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "알 수 없는 인자: $1"
            echo "사용법: bash data/finish_korean_pipeline.sh [--from-step N] [--log-file PATH] [--dry-run]"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# 로그 설정: 이후 모든 stdout/stderr를 파일 + 터미널로 동시 출력
# (--dry-run 시에도 로그 파일 생성)
# -----------------------------------------------------------------------------
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

# -----------------------------------------------------------------------------
# 유틸리티 함수
# -----------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_sep() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ================================================================"
}

log_skip() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] $*"
}

log_start() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [START] $*"
}

log_done() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DONE] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

# 명령 실행 (dry-run 시 출력만)
# PYTHONUNBUFFERED=1: Python stdout 즉시 flush → tee 경유 로그 파일에 실시간 반영
run_cmd() {
    if $DRY_RUN; then
        echo "[DRY-RUN] $*"
    else
        PYTHONUNBUFFERED=1 "$@"
    fi
}

# 파일 크기를 사람이 읽기 쉬운 형식으로 출력
human_size() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        echo "N/A"
        return
    fi
    local bytes
    bytes=$(stat -c%s "${file}" 2>/dev/null || echo 0)
    if   (( bytes >= 1073741824 )); then
        awk "BEGIN { printf \"%.2f GB\", ${bytes}/1073741824 }"
    elif (( bytes >= 1048576 )); then
        awk "BEGIN { printf \"%.2f MB\", ${bytes}/1048576 }"
    elif (( bytes >= 1024 )); then
        awk "BEGIN { printf \"%.2f KB\", ${bytes}/1024 }"
    else
        echo "${bytes} B"
    fi
}

# .bin 파일의 토큰 수 추정 (uint16 = 2바이트/토큰)
token_count() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        echo "N/A"
        return
    fi
    local bytes
    bytes=$(stat -c%s "${file}" 2>/dev/null || echo 0)
    local tokens=$(( bytes / 2 ))
    if   (( tokens >= 1000000000 )); then
        awk "BEGIN { printf \"%.2fB\", ${tokens}/1000000000 }"
    elif (( tokens >= 1000000 )); then
        awk "BEGIN { printf \"%.2fM\", ${tokens}/1000000 }"
    elif (( tokens >= 1000 )); then
        awk "BEGIN { printf \"%.2fK\", ${tokens}/1000 }"
    else
        echo "${tokens}"
    fi
}

# 스텝 실행 여부 결정
# 인자: step_num output_file
# 반환: 0 = 실행해야 함, 1 = 건너뜀
should_skip() {
    local step_num="$1"
    local output_file="$2"

    # --from-step 이 지정되어 있고, 현재 스텝이 그 이상이면 강제 실행
    if (( FROM_STEP > 0 && step_num >= FROM_STEP )); then
        return 1  # 건너뛰지 않음 (실행)
    fi

    # 출력 파일이 이미 존재하면 건너뜀
    if [[ -f "${output_file}" ]]; then
        return 0  # 건너뜀
    fi

    return 1  # 실행
}

# -----------------------------------------------------------------------------
# 경로 상수
# -----------------------------------------------------------------------------
TOKENIZER="tokenizer/korean_sp/tokenizer.json"

RAW_C4="data/raw/c4_ko"
RAW_NAMU="data/raw/namuwiki_ko"
RAW_WIKI_PATTERN="data/raw/ko_wiki_*.txt"

OUT_C4_TRAIN="data/korean_c4_train.bin"
OUT_C4_VAL="data/korean_c4_val.bin"

OUT_NAMU_TRAIN="data/korean_namuwiki_train.bin"
OUT_NAMU_VAL="data/korean_namuwiki_val.bin"

OUT_WIKI_TRAIN="data/korean_wiki_train.bin"
OUT_WIKI_VAL="data/korean_wiki_val.bin"

OUT_TRAIN="data/korean_train.bin"
OUT_VAL="data/korean_val.bin"

# -----------------------------------------------------------------------------
# 시작 메시지
# -----------------------------------------------------------------------------
log_sep
log "한국어 LLM 데이터 파이프라인 (Step 6~7) 재개"
log "프로젝트 루트 : ${PROJECT_ROOT}"
log "로그 파일     : ${LOG_FILE}"
log "FROM_STEP     : ${FROM_STEP} (0=자동감지)"
log "DRY_RUN       : ${DRY_RUN}"
log_sep

# -----------------------------------------------------------------------------
# 사전 검사
# -----------------------------------------------------------------------------
log "사전 검사 시작..."

# 토크나이저 존재 확인
if [[ ! -f "${TOKENIZER}" ]]; then
    log_error "토크나이저를 찾을 수 없습니다: ${TOKENIZER}"
    exit 1
fi
log "토크나이저 확인: ${TOKENIZER} ($(human_size "${TOKENIZER}"))"

# CC-100은 비어있으므로 건너뜀 알림
if [[ -d "data/raw/cc100_ko" ]]; then
    local_files=$(find "data/raw/cc100_ko" -type f 2>/dev/null | wc -l)
    if (( local_files == 0 )); then
        log "CC-100: data/raw/cc100_ko 디렉토리가 비어있음 → CC-100 처리 건너뜀"
    fi
fi

# 입력 데이터 존재 확인
c4_files=$(find "${RAW_C4}" -name "*.txt" -type f 2>/dev/null | wc -l)
namu_files=$(find "${RAW_NAMU}" -name "*.txt" -type f 2>/dev/null | wc -l)
wiki_files=$(find "data/raw" -name "ko_wiki_*.txt" -type f 2>/dev/null | wc -l)

log "입력 데이터 현황:"
log "  c4_ko      : ${c4_files}개 .txt 파일 (${RAW_C4})"
log "  namuwiki_ko: ${namu_files}개 .txt 파일 (${RAW_NAMU})"
log "  ko_wiki    : ${wiki_files}개 .txt 파일 (data/raw/ko_wiki_*.txt)"

if (( c4_files == 0 )); then
    log_error "c4_ko 데이터 없음: ${RAW_C4} 에 .txt 파일이 없습니다"
    exit 1
fi
if (( namu_files == 0 )); then
    log_error "namuwiki 데이터 없음: ${RAW_NAMU} 에 .txt 파일이 없습니다"
    exit 1
fi
if (( wiki_files == 0 )); then
    log_error "ko_wiki 데이터 없음: data/raw/ko_wiki_*.txt 파일이 없습니다"
    exit 1
fi

log "사전 검사 완료"
log_sep

# =============================================================================
# Step 6a: c4_ko 토크나이징
# =============================================================================
STEP_NUM=61

if should_skip ${STEP_NUM} "${OUT_C4_TRAIN}"; then
    log_skip "Step 6a (c4_ko 토크나이징): ${OUT_C4_TRAIN} 이미 존재 → 건너뜀"
    log "       크기: $(human_size "${OUT_C4_TRAIN}"), 토큰: $(token_count "${OUT_C4_TRAIN}")"
else
    log_start "Step 6a: c4_ko 토크나이징 시작"
    log "  입력: ${RAW_C4}/*.txt (${c4_files}개 파일)"
    log "  출력: ${OUT_C4_TRAIN}, ${OUT_C4_VAL}"
    log "  토크나이저: ${TOKENIZER}"

    # 강제 재실행 시 기존 파일 제거
    if (( FROM_STEP > 0 && STEP_NUM >= FROM_STEP )); then
        if [[ -f "${OUT_C4_TRAIN}" ]]; then
            log "  기존 파일 삭제 (강제 재실행): ${OUT_C4_TRAIN}"
            run_cmd rm -f "${OUT_C4_TRAIN}" "${OUT_C4_VAL}"
        fi
    fi

    STEP6A_START=$(date +%s)
    run_cmd python data/prepare.py \
        --input "${RAW_C4}/*.txt" \
        --output "${OUT_C4_TRAIN}" \
        --tokenizer "${TOKENIZER}" \
        --val_split 0.002 \
        --seed 42

    if ! $DRY_RUN; then
        STEP6A_END=$(date +%s)
        STEP6A_ELAPSED=$(( STEP6A_END - STEP6A_START ))
        log_done "Step 6a 완료 (소요: ${STEP6A_ELAPSED}초)"
        log "  ${OUT_C4_TRAIN} : $(human_size "${OUT_C4_TRAIN}"), 토큰: $(token_count "${OUT_C4_TRAIN}")"
        log "  ${OUT_C4_VAL}   : $(human_size "${OUT_C4_VAL}"), 토큰: $(token_count "${OUT_C4_VAL}")"
    else
        log_done "Step 6a (dry-run 완료)"
    fi
fi

log_sep

# =============================================================================
# Step 6b: namuwiki_ko 토크나이징
# =============================================================================
STEP_NUM=62

if should_skip ${STEP_NUM} "${OUT_NAMU_TRAIN}"; then
    log_skip "Step 6b (namuwiki 토크나이징): ${OUT_NAMU_TRAIN} 이미 존재 → 건너뜀"
    log "       크기: $(human_size "${OUT_NAMU_TRAIN}"), 토큰: $(token_count "${OUT_NAMU_TRAIN}")"
else
    log_start "Step 6b: namuwiki_ko 토크나이징 시작"
    log "  입력: ${RAW_NAMU}/*.txt (${namu_files}개 파일)"
    log "  출력: ${OUT_NAMU_TRAIN}, ${OUT_NAMU_VAL}"
    log "  토크나이저: ${TOKENIZER}"

    # 강제 재실행 시 기존 파일 제거
    if (( FROM_STEP > 0 && STEP_NUM >= FROM_STEP )); then
        if [[ -f "${OUT_NAMU_TRAIN}" ]]; then
            log "  기존 파일 삭제 (강제 재실행): ${OUT_NAMU_TRAIN}"
            run_cmd rm -f "${OUT_NAMU_TRAIN}" "${OUT_NAMU_VAL}"
        fi
    fi

    STEP6B_START=$(date +%s)
    run_cmd python data/prepare.py \
        --input "${RAW_NAMU}/*.txt" \
        --output "${OUT_NAMU_TRAIN}" \
        --tokenizer "${TOKENIZER}" \
        --val_split 0.002 \
        --seed 42

    if ! $DRY_RUN; then
        STEP6B_END=$(date +%s)
        STEP6B_ELAPSED=$(( STEP6B_END - STEP6B_START ))
        log_done "Step 6b 완료 (소요: ${STEP6B_ELAPSED}초)"
        log "  ${OUT_NAMU_TRAIN} : $(human_size "${OUT_NAMU_TRAIN}"), 토큰: $(token_count "${OUT_NAMU_TRAIN}")"
        log "  ${OUT_NAMU_VAL}   : $(human_size "${OUT_NAMU_VAL}"), 토큰: $(token_count "${OUT_NAMU_VAL}")"
    else
        log_done "Step 6b (dry-run 완료)"
    fi
fi

log_sep

# =============================================================================
# Step 6c: ko_wiki 토크나이징
# =============================================================================
STEP_NUM=63

if should_skip ${STEP_NUM} "${OUT_WIKI_TRAIN}"; then
    log_skip "Step 6c (ko_wiki 토크나이징): ${OUT_WIKI_TRAIN} 이미 존재 → 건너뜀"
    log "       크기: $(human_size "${OUT_WIKI_TRAIN}"), 토큰: $(token_count "${OUT_WIKI_TRAIN}")"
else
    log_start "Step 6c: ko_wiki 토크나이징 시작"
    log "  입력: data/raw/ko_wiki_*.txt (${wiki_files}개 파일)"
    log "  출력: ${OUT_WIKI_TRAIN}, ${OUT_WIKI_VAL}"
    log "  토크나이저: ${TOKENIZER}"

    # 강제 재실행 시 기존 파일 제거
    if (( FROM_STEP > 0 && STEP_NUM >= FROM_STEP )); then
        if [[ -f "${OUT_WIKI_TRAIN}" ]]; then
            log "  기존 파일 삭제 (강제 재실행): ${OUT_WIKI_TRAIN}"
            run_cmd rm -f "${OUT_WIKI_TRAIN}" "${OUT_WIKI_VAL}"
        fi
    fi

    STEP6C_START=$(date +%s)
    run_cmd python data/prepare.py \
        --input "data/raw/ko_wiki_*.txt" \
        --output "${OUT_WIKI_TRAIN}" \
        --tokenizer "${TOKENIZER}" \
        --val_split 0.002 \
        --seed 42

    if ! $DRY_RUN; then
        STEP6C_END=$(date +%s)
        STEP6C_ELAPSED=$(( STEP6C_END - STEP6C_START ))
        log_done "Step 6c 완료 (소요: ${STEP6C_ELAPSED}초)"
        log "  ${OUT_WIKI_TRAIN} : $(human_size "${OUT_WIKI_TRAIN}"), 토큰: $(token_count "${OUT_WIKI_TRAIN}")"
        log "  ${OUT_WIKI_VAL}   : $(human_size "${OUT_WIKI_VAL}"), 토큰: $(token_count "${OUT_WIKI_VAL}")"
    else
        log_done "Step 6c (dry-run 완료)"
    fi
fi

log_sep

# =============================================================================
# Step 7: 병합 (korean_train.bin / korean_val.bin)
# =============================================================================
STEP_NUM=70

if should_skip ${STEP_NUM} "${OUT_TRAIN}"; then
    log_skip "Step 7 (병합): ${OUT_TRAIN} 이미 존재 → 건너뜀"
    log "       크기: $(human_size "${OUT_TRAIN}"), 토큰: $(token_count "${OUT_TRAIN}")"
else
    log_start "Step 7: 병합 시작"

    # 병합 대상 파일 확인 (dry-run이 아닐 경우에만 존재 확인)
    if ! $DRY_RUN; then
        MISSING_TRAINS=()
        for f in "${OUT_C4_TRAIN}" "${OUT_NAMU_TRAIN}" "${OUT_WIKI_TRAIN}"; do
            if [[ ! -f "${f}" ]]; then
                MISSING_TRAINS+=("${f}")
            fi
        done
        if (( ${#MISSING_TRAINS[@]} > 0 )); then
            log_error "병합에 필요한 train 파일이 없습니다:"
            for f in "${MISSING_TRAINS[@]}"; do
                log_error "  - ${f}"
            done
            exit 1
        fi

        MISSING_VALS=()
        for f in "${OUT_C4_VAL}" "${OUT_NAMU_VAL}" "${OUT_WIKI_VAL}"; do
            if [[ ! -f "${f}" ]]; then
                MISSING_VALS+=("${f}")
            fi
        done
        if (( ${#MISSING_VALS[@]} > 0 )); then
            log_error "병합에 필요한 val 파일이 없습니다:"
            for f in "${MISSING_VALS[@]}"; do
                log_error "  - ${f}"
            done
            exit 1
        fi
    fi

    # 강제 재실행 시 기존 병합 파일 제거
    if (( FROM_STEP > 0 && STEP_NUM >= FROM_STEP )); then
        if [[ -f "${OUT_TRAIN}" ]]; then
            log "  기존 파일 삭제 (강제 재실행): ${OUT_TRAIN}"
            run_cmd rm -f "${OUT_TRAIN}" "${OUT_VAL}"
        fi
    fi

    log "  [train] 병합:"
    log "    입력: ${OUT_C4_TRAIN}, ${OUT_NAMU_TRAIN}, ${OUT_WIKI_TRAIN}"
    log "    출력: ${OUT_TRAIN}"

    STEP7_START=$(date +%s)
    run_cmd python data/merge_bins.py \
        "${OUT_C4_TRAIN}" \
        "${OUT_NAMU_TRAIN}" \
        "${OUT_WIKI_TRAIN}" \
        "${OUT_TRAIN}"

    log "  [val] 병합:"
    log "    입력: ${OUT_C4_VAL}, ${OUT_NAMU_VAL}, ${OUT_WIKI_VAL}"
    log "    출력: ${OUT_VAL}"

    run_cmd python data/merge_bins.py \
        "${OUT_C4_VAL}" \
        "${OUT_NAMU_VAL}" \
        "${OUT_WIKI_VAL}" \
        "${OUT_VAL}"

    if ! $DRY_RUN; then
        STEP7_END=$(date +%s)
        STEP7_ELAPSED=$(( STEP7_END - STEP7_START ))
        log_done "Step 7 완료 (소요: ${STEP7_ELAPSED}초)"
        log "  ${OUT_TRAIN} : $(human_size "${OUT_TRAIN}"), 토큰: $(token_count "${OUT_TRAIN}")"
        log "  ${OUT_VAL}   : $(human_size "${OUT_VAL}"), 토큰: $(token_count "${OUT_VAL}")"
    else
        log_done "Step 7 (dry-run 완료)"
    fi
fi

log_sep

# =============================================================================
# 최종 상태 요약
# =============================================================================
log "=== 파이프라인 완료 요약 ==="

print_file_info() {
    local label="$1"
    local file="$2"
    if [[ -f "${file}" ]]; then
        printf "[$(date '+%Y-%m-%d %H:%M:%S')] %-45s  크기: %10s  토큰: %10s\n" \
            "${label}" "$(human_size "${file}")" "$(token_count "${file}")"
    else
        printf "[$(date '+%Y-%m-%d %H:%M:%S')] %-45s  [파일 없음]\n" "${label}"
    fi
}

print_file_info "korean_c4_train.bin"       "${OUT_C4_TRAIN}"
print_file_info "korean_c4_val.bin"         "${OUT_C4_VAL}"
print_file_info "korean_namuwiki_train.bin" "${OUT_NAMU_TRAIN}"
print_file_info "korean_namuwiki_val.bin"   "${OUT_NAMU_VAL}"
print_file_info "korean_wiki_train.bin"     "${OUT_WIKI_TRAIN}"
print_file_info "korean_wiki_val.bin"       "${OUT_WIKI_VAL}"
print_file_info "korean_train.bin [최종]"   "${OUT_TRAIN}"
print_file_info "korean_val.bin   [최종]"   "${OUT_VAL}"

# 총 학습 토큰 계산
if [[ -f "${OUT_TRAIN}" ]] && ! $DRY_RUN; then
    TRAIN_BYTES=$(stat -c%s "${OUT_TRAIN}" 2>/dev/null || echo 0)
    TRAIN_TOKENS=$(( TRAIN_BYTES / 2 ))
    TRAIN_TOKENS_B=$(awk "BEGIN { printf \"%.2fB\", ${TRAIN_TOKENS}/1000000000 }")
    log ""
    log "총 학습 토큰: ${TRAIN_TOKENS_B}  (${TRAIN_TOKENS} tokens)"
fi

log_sep
log "모든 단계 완료"

# =============================================================================
# 실행 안내 (스크립트 첫 실행 시에도 볼 수 있도록 출력)
# =============================================================================
cat <<'EOF'

실행 방법:
  # 자동 감지 (완료된 스텝 건너뜀)
  bash data/finish_korean_pipeline.sh

  # 백그라운드 실행 (권장)
  nohup bash data/finish_korean_pipeline.sh > data/finish_korean_pipeline.log 2>&1 &
  tail -f data/finish_korean_pipeline.log

  # 특정 스텝부터 재시작
  bash data/finish_korean_pipeline.sh --from-step 62

  # dry-run (실제 실행 없이 명령 확인)
  bash data/finish_korean_pipeline.sh --dry-run
EOF
