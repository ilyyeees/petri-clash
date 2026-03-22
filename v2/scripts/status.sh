#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "processes"
ps -eo pid,etimes,%cpu,%mem,cmd | grep -E "v2\\.train_all_v2|v2\\.train_v2|v2\\.preflight" | grep -v "grep -E" || true

LATEST_LOG="$(ls -1t runs_v2/logs/*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "$LATEST_LOG" ]]; then
  echo
  echo "latest log: $LATEST_LOG"
  tail -n 40 "$LATEST_LOG"
fi
