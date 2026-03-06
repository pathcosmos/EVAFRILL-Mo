#!/usr/bin/env bash
#
# clickhouse-watchdog.sh — ClickHouse 헬스체크 + 자동 재시작
# crontab에 등록하여 1분마다 실행
#
# Usage:
#   */1 * * * * /PROJECT/0325120031_A/ghong/taketimes/llm-bang/scripts/clickhouse-watchdog.sh
#

set -euo pipefail

# ── 설정 ──────────────────────────────────────────────
CH_BIN="/PROJECT/0325120031_A/ghong/taketimes/clickhouse-bin"
CH_CONFIG="/PROJECT/0325120031_A/ghong/taketimes/llm-bang/configs/clickhouse-config.xml"
TCP_PORT=9000
HTTP_PORT=8123
HOST="127.0.0.1"

LOG_DIR="/tmp/clickhouse"
LOG_FILE="${LOG_DIR}/watchdog.log"
MAX_LOG_SIZE=$((10 * 1024 * 1024))  # 10MB 로테이션

RESTART_COOLDOWN=180  # 초 — 재시작 후 이 시간 내 재시도 방지
LAST_RESTART_FILE="/tmp/clickhouse-last-restart"
HEALTH_CHECK_TIMEOUT=5  # 초 — 헬스체크 curl/query 타임아웃

# ── 함수 ──────────────────────────────────────────────
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [clickhouse-watchdog] $*" >> "$LOG_FILE"
}

rotate_log() {
    local file="$1"
    if [[ -f "$file" ]] && [[ $(stat -c%s "$file" 2>/dev/null || echo 0) -gt $MAX_LOG_SIZE ]]; then
        mv "$file" "${file}.old"
        log "Log rotated: $file"
    fi
}

is_tcp_port_open() {
    if command -v ss &>/dev/null; then
        ss -tlnH "sport = :${TCP_PORT}" 2>/dev/null | grep -q "$TCP_PORT"
    else
        (echo > /dev/tcp/"$HOST"/"$TCP_PORT") 2>/dev/null
    fi
}

is_http_responding() {
    # HTTP 인터페이스 핑 — ClickHouse는 GET / 에 "Ok.\n" 응답
    if command -v curl &>/dev/null; then
        local resp
        resp=$(curl -s --max-time "$HEALTH_CHECK_TIMEOUT" "http://${HOST}:${HTTP_PORT}/ping" 2>/dev/null || true)
        [[ "$resp" == "Ok." ]]
    else
        # curl 없으면 TCP 포트만 확인
        (echo > /dev/tcp/"$HOST"/"$HTTP_PORT") 2>/dev/null
    fi
}

is_process_alive() {
    # ClickHouse 내부 watchdog 프로세스명: "clickhouse-watchdog" (바이너리 자체)
    # 이 스크립트(clickhouse-watchdog.sh)와 구분하기 위해 --daemon 플래그 포함 패턴 사용
    pgrep -f "clickhouse.*server.*--daemon" >/dev/null 2>&1
}

can_execute_query() {
    # 실제 쿼리 실행으로 서버가 응답하는지 확인
    local result
    result=$("$CH_BIN" client --port "$TCP_PORT" --query "SELECT 1" 2>/dev/null || true)
    [[ "$result" == "1" ]]
}

cooldown_active() {
    if [[ -f "$LAST_RESTART_FILE" ]]; then
        local last_restart now diff
        last_restart=$(cat "$LAST_RESTART_FILE" 2>/dev/null)
        now=$(date +%s)
        diff=$(( now - last_restart ))
        if [[ $diff -lt $RESTART_COOLDOWN ]]; then
            return 0  # 쿨다운 중
        fi
    fi
    return 1  # 쿨다운 아님
}

stop_existing() {
    log "Stopping existing ClickHouse processes..."
    local my_pid=$$
    local pids

    # 정상 종료 시도 (서버 프로세스)
    pids=$(pgrep -f "clickhouse.*server.*--daemon" 2>/dev/null | grep -v "^${my_pid}$" || true)
    if [[ -n "$pids" ]]; then
        log "Sending TERM to PIDs: $pids"
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 3
        # 아직 살아있으면 강제 종료
        pids=$(pgrep -f "clickhouse.*server.*--daemon" 2>/dev/null | grep -v "^${my_pid}$" || true)
        if [[ -n "$pids" ]]; then
            log "Force killing PIDs: $pids"
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 2
        fi
    fi
}

start_server() {
    log "Starting ClickHouse server (daemon mode)..."

    # 기존 프로세스 정리
    stop_existing

    # 필요한 디렉토리 생성
    mkdir -p /tmp/clickhouse/logs
    mkdir -p /tmp/clickhouse-tmp

    # 데몬 모드로 시작
    "$CH_BIN" server --config-file="$CH_CONFIG" --daemon

    # 시작 후 대기 + 확인 (최대 15초)
    local attempts=0
    local max_attempts=15
    while [[ $attempts -lt $max_attempts ]]; do
        sleep 1
        attempts=$((attempts + 1))
        if is_tcp_port_open && can_execute_query; then
            date +%s > "$LAST_RESTART_FILE"
            log "ClickHouse started successfully (took ${attempts}s)"
            return 0
        fi
    done

    date +%s > "$LAST_RESTART_FILE"
    log "ERROR: ClickHouse did not respond within ${max_attempts}s after start"
    return 1
}

# ── 메인 로직 ─────────────────────────────────────────
rotate_log "$LOG_FILE"

# 1) 바이너리 존재 확인
if [[ ! -x "$CH_BIN" ]]; then
    log "FATAL: ClickHouse binary not found or not executable: $CH_BIN"
    exit 1
fi

# 2) 프로세스 + 포트 + 쿼리 체크
process_ok=false
port_ok=false
query_ok=false

if is_process_alive; then
    process_ok=true
fi

if is_tcp_port_open; then
    port_ok=true
fi

if $port_ok && can_execute_query; then
    query_ok=true
fi

# 3) 판단
if $process_ok && $port_ok && $query_ok; then
    # 완전 정상 — 아무것도 안 함
    exit 0
fi

# HTTP도 확인 (진단 로그용)
http_ok=false
if is_http_responding; then
    http_ok=true
fi

# 비정상 상태 로깅
if $process_ok && $port_ok && ! $query_ok; then
    log "WARN: Process alive, port open, but query failed. Possible hung state."
elif $process_ok && ! $port_ok; then
    log "WARN: Process alive but TCP port $TCP_PORT not listening."
elif ! $process_ok; then
    log "WARN: ClickHouse is completely down (no process found)."
fi
log "Status: process=$process_ok port=$port_ok query=$query_ok http=$http_ok"

# 4) 쿨다운 체크
if cooldown_active; then
    log "Cooldown active (last restart < ${RESTART_COOLDOWN}s ago). Skipping."
    exit 0
fi

# 5) 재시작
log "Attempting ClickHouse restart..."
if start_server; then
    log "ClickHouse restart SUCCESS"
else
    log "ClickHouse restart FAILED"
    exit 1
fi
