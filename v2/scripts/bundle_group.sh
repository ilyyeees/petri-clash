#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${1:-v2/configs/rtx5090_base.toml}"
shift || true

python3 -m v2.bundle_run --config "$CONFIG" "$@"
