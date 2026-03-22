#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${1:-v2/configs/rtx5090_base.toml}"
shift || true

mkdir -p runs_v2/logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="runs_v2/logs/train_all-${STAMP}.log"
PID="runs_v2/logs/train_all-${STAMP}.pid"

nohup env PYTHONUNBUFFERED=1 python3 v2/train_all_v2.py --config "$CONFIG" "$@" > "$LOG" 2>&1 &
echo $! | tee "$PID"
echo "pid file: $PID"
echo "log file: $LOG"
