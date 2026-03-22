#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "processes"
ps -eo pid,etimes,%cpu,%mem,cmd | grep -E "trainer\\.train_all_v2|trainer\\.train_v2|trainer\\.preflight" | grep -v "grep -E" || true

LATEST_LOG="$(ls -1t runs_v2/logs/*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "$LATEST_LOG" ]]; then
  echo
  echo "latest log: $LATEST_LOG"
  tail -n 40 "$LATEST_LOG"
fi
