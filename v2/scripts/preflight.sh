#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${1:-v2/configs/rtx5090_base.toml}"
TARGET="${2:-}"

if [[ -n "$TARGET" ]]; then
  PYTHONUNBUFFERED=1 python3 -m v2.preflight --config "$CONFIG" --target "$TARGET"
else
  PYTHONUNBUFFERED=1 python3 -m v2.preflight --config "$CONFIG"
fi
