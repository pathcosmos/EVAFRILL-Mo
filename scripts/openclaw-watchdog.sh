#!/usr/bin/env bash
#
# openclaw-watchdog.sh — OpenClaw Gateway 헬스체크 + 자동 재시작
# crontab에 등록하여 1분마다 실행
#
# Usage:
#   */1 * * * * /PROJECT/0325120031_A/ghong/taketimes/llm-bang/scripts/openclaw-watchdog.sh
#
# 변경이력:
#   2026-03-01  네트워크 체크를 ICMP→HTTP로 변경 (ICMP 차단 환경 대응)
#              다중 엔드포인트 fallback, 게이트웨이 HTTP 응답 체크 추가
#              setsid 분리 실행, 상세 로깅 강화

set -euo pipefail

# ── 설정 ──────────────────────────────────────────────
RNTIER_HOME="REDACTED_RNTIER_PATH"
OPENCLAW_BIN="${RNTIER_HOME}/.npm-global/bin/openclaw"
GATEWAY_PORT=18789
GATEWAY_HOST="127.0.0.1"
PID_FILE="/tmp/openclaw-gateway.pid"
LOG_DIR="/tmp/openclaw"
LOG_FILE="${LOG_DIR}/watchdog.log"
GATEWAY_LOG="${LOG_DIR}/gateway.log"
MAX_LOG_SIZE=$((10 * 1024 * 1024))  # 10MB 로테이션
RESTART_COOLDOWN=120  # 초 — 재시작 후 이 시간 내 재시도 방지
LAST_RESTART_FILE="/tmp/openclaw-last-restart"
CONSECUTIVE_FAIL_FILE="/tmp/openclaw-consecutive-fails"

# 환경변수 — openclaw가 config를 찾을 수 있도록
export PATH="${RNTIER_HOME}/.npm-global/bin:/usr/bin:/usr/local/bin:/bin:$PATH"
export HOME="/home/ghong"
export OPENCLAW_STATE_DIR="${RNTIER_HOME}/.openclaw"
export OPENCLAW_CONFIG_PATH="${RNTIER_HOME}/.openclaw/openclaw.json"

# ── 함수 ──────────────────────────────────────────────
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

rotate_log() {
    local file="$1"
    if [[ -f "$file" ]] && [[ $(stat -c%s "$file" 2>/dev/null || echo 0) -gt $MAX_LOG_SIZE ]]; then
        mv "$file" "${file}.old"
        log "Log rotated: $file"
    fi
}

# 게이트웨이의 실제 엔드포인트로 로컬 HTTP 응답 체크
check_gateway_http() {
    if command -v curl &>/dev/null; then
        curl -sf --max-time 5 -o /dev/null "http://${GATEWAY_HOST}:${GATEWAY_PORT}/__openclaw__/canvas/" 2>/dev/null
        return $?
    fi
    return 1
}

is_port_open() {
    if command -v ss &>/dev/null; then
        ss -tlnH "sport = :${GATEWAY_PORT}" 2>/dev/null | grep -q "$GATEWAY_PORT"
    else
        (echo > /dev/tcp/"$GATEWAY_HOST"/"$GATEWAY_PORT") 2>/dev/null
    fi
}

is_process_alive() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    pgrep -f "openclaw.*gateway" >/dev/null 2>&1
}

# 네트워크 체크 — DNS 해석 기반
# 이 서버는 ICMP(ping)과 아웃바운드 HTTPS(curl)가 모두 차단됨.
# 단, DNS 해석은 가능하고 게이트웨이(Node.js)는 long-polling으로 통신 가능.
# 따라서 DNS 해석 성공 여부로 "네트워크 자체가 살아있는지" 판단한다.
check_network() {
    # 방법1: getent (가장 빠르고 가벼움)
    if command -v getent &>/dev/null; then
        getent hosts api.telegram.org >/dev/null 2>&1 && return 0
        getent hosts api.anthropic.com >/dev/null 2>&1 && return 0
    fi
    # 방법2: nslookup
    if command -v nslookup &>/dev/null; then
        nslookup -timeout=5 api.telegram.org >/dev/null 2>&1 && return 0
    fi
    # 방법3: /dev/tcp 로 DNS 서버(168.126.63.1) 포트 53 확인
    (echo > /dev/tcp/168.126.63.1/53) 2>/dev/null && return 0
    return 1
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

get_consecutive_fails() {
    if [[ -f "$CONSECUTIVE_FAIL_FILE" ]]; then
        cat "$CONSECUTIVE_FAIL_FILE" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

set_consecutive_fails() {
    echo "$1" > "$CONSECUTIVE_FAIL_FILE"
}

start_gateway() {
    log "ACTION: Starting OpenClaw gateway on port $GATEWAY_PORT..."

    # 기존 좀비 프로세스 정리
    local old_pids
    old_pids=$(pgrep -f "openclaw.*gateway" 2>/dev/null || true)
    if [[ -n "$old_pids" ]]; then
        log "ACTION: Killing stale gateway processes: $old_pids"
        echo "$old_pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    # 게이트웨이 시작 — setsid로 완전 분리 (부모 프로세스 시그널 전파 방지)
    setsid nohup "$OPENCLAW_BIN" gateway run \
        --port "$GATEWAY_PORT" \
        --bind loopback \
        >> "$GATEWAY_LOG" 2>&1 < /dev/null &

    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    date +%s > "$LAST_RESTART_FILE"

    log "ACTION: Gateway launched with PID $new_pid (setsid)"

    # 8초 대기 후 확인 (Telegram provider 초기화에 시간 필요)
    sleep 8
    if kill -0 "$new_pid" 2>/dev/null; then
        log "OK: Gateway PID $new_pid is alive after startup"
        if is_port_open; then
            log "OK: Port $GATEWAY_PORT is listening"
        else
            log "WARN: Gateway alive but port $GATEWAY_PORT not yet listening (may need more time)"
        fi
        return 0
    else
        log "ERROR: Gateway PID $new_pid died immediately after start"
        log "ERROR: Last 10 lines of gateway.log:"
        tail -10 "$GATEWAY_LOG" 2>/dev/null | while read -r line; do
            log "  | $line"
        done
        return 1
    fi
}

# ── 메인 로직 ─────────────────────────────────────────
rotate_log "$LOG_FILE"
rotate_log "$GATEWAY_LOG"

# 오래된 openclaw 로그 파일 정리 (7일 이상)
find "$LOG_DIR" -name "openclaw-*.log" -mtime +7 -delete 2>/dev/null || true

# 1) 프로세스 + 포트 체크를 먼저 수행 (게이트웨이가 살아있으면 네트워크 체크 불필요)
process_ok=false
port_ok=false
http_ok=false

if is_process_alive; then
    process_ok=true
fi

if is_port_open; then
    port_ok=true
fi

if $port_ok && check_gateway_http; then
    http_ok=true
fi

# 2) 게이트웨이 정상이면 바로 종료
if $process_ok && $port_ok; then
    if $http_ok; then
        # 완전 정상
        set_consecutive_fails 0
        exit 0
    fi
    # 프로세스+포트 OK인데 HTTP 응답 없음 → hung 가능성
    fails=$(get_consecutive_fails)
    fails=$((fails + 1))
    set_consecutive_fails "$fails"
    log "WARN: Process alive, port open, but HTTP not responding (consecutive: $fails)"
    if [[ $fails -lt 3 ]]; then
        log "INFO: Waiting more cycles before restart (transient check, $fails/3)"
        exit 0
    fi
    log "WARN: HTTP unresponsive for $fails consecutive checks — proceeding to restart"
fi

# 3) 게이트웨이가 비정상 — 네트워크 체크 후 재시작 여부 판단
if $process_ok && ! $port_ok; then
    log "WARN: Process alive but port $GATEWAY_PORT not listening. Possible hung state."
fi

if ! $process_ok && ! $port_ok; then
    log "WARN: Gateway is completely down (no process, no port)."
fi

if ! $process_ok && $port_ok; then
    log "WARN: No known gateway process but port $GATEWAY_PORT is in use. Stale process?"
fi

# 4) 네트워크 체크 — DNS 기반 (게이트웨이가 죽었을 때만 실행)
if ! check_network; then
    log "WARN: Network unreachable (DNS resolution failed). Skipping gateway restart."
    exit 0
fi

# 5) 쿨다운 체크
if cooldown_active; then
    log "INFO: Cooldown active (last restart < ${RESTART_COOLDOWN}s ago). Skipping."
    exit 0
fi

# 6) 재시작
log "ACTION: Attempting gateway restart..."
if start_gateway; then
    log "OK: Gateway restart SUCCESS"
    set_consecutive_fails 0
else
    log "ERROR: Gateway restart FAILED"
    exit 1
fi
