#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${1:-trainer/configs/single_gpu_base.toml}"
TARGET="${2:-}"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

if [[ -n "$TARGET" ]]; then
  PYTHONUNBUFFERED=1 "$PYTHON" -m trainer.preflight --config "$CONFIG" --target "$TARGET"
else
  PYTHONUNBUFFERED=1 "$PYTHON" -m trainer.preflight --config "$CONFIG"
fi
