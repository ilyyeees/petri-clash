#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${1:-v2/configs/single_gpu_base.toml}"
shift || true

mkdir -p runs_v2/logs
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="runs_v2/logs/train_all-${STAMP}.log"

PYTHONUNBUFFERED=1 python3 -m v2.train_all_v2 --config "$CONFIG" "$@" | tee "$LOG"
