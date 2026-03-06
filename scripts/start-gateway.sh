#!/usr/bin/env bash
# start-gateway.sh — OpenClaw 게이트웨이 직접 시작 (독립 프로세스)
set -euo pipefail

RNTIER_HOME="${RNTIER_HOME:-$HOME}"
export PATH="${RNTIER_HOME}/.npm-global/bin:/usr/bin:/usr/local/bin:/bin:$PATH"
export HOME="${HOME:-/home/ghong}"
export OPENCLAW_STATE_DIR="${RNTIER_HOME}/.openclaw"
export OPENCLAW_CONFIG_PATH="${RNTIER_HOME}/.openclaw/openclaw.json"

LOG_DIR="/tmp/openclaw"
GATEWAY_LOG="${LOG_DIR}/gateway.log"
PID_FILE="/tmp/openclaw-gateway.pid"

mkdir -p "$LOG_DIR"

# 기존 프로세스 정리
pkill -f "openclaw.*gateway" 2>/dev/null || true
sleep 2

# 게이트웨이 시작 — setsid로 완전 분리
setsid nohup "${RNTIER_HOME}/.npm-global/bin/openclaw" gateway run \
    --port 18789 \
    --bind loopback \
    >> "$GATEWAY_LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
date +%s > /tmp/openclaw-last-restart

echo "[$(date)] Gateway launched with PID $PID"

# 10초 대기 후 상태 확인
sleep 10

if kill -0 "$PID" 2>/dev/null; then
    echo "[$(date)] OK: Gateway PID $PID is alive"
    ss -tlnH "sport = :18789" 2>/dev/null && echo "[$(date)] OK: Port 18789 is listening" || echo "[$(date)] WARN: Port 18789 not yet listening"
else
    echo "[$(date)] FAIL: Gateway PID $PID died"
    echo "--- Last 20 lines of gateway.log ---"
    tail -20 "$GATEWAY_LOG" 2>/dev/null
    exit 1
fi
