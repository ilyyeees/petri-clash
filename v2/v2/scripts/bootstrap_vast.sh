#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
PYTHON="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

"$PIP" install --upgrade pip setuptools wheel

if ! "$PYTHON" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  "$PIP" install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
fi

"$PIP" install -r v2/requirements.txt

"$PYTHON" - <<'PY'
try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "torch is not installed in this image. use Vast's pytorch template with the automatic image tag."
    ) from exc

def parse_version(text):
    base = text.split("+", 1)[0]
    return tuple(int(part) for part in base.split(".")[:2])

torch_version = parse_version(torch.__version__)
cuda_version = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])

print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))

if not torch.cuda.is_available():
    raise SystemExit("cuda is not available. pick a cuda-enabled vast template before training.")

if torch_version < (2, 7):
    raise SystemExit("torch is too old for this training stack. use torch 2.7 or newer.")

if cuda_version < (12, 6):
    raise SystemExit("cuda build is too old for this training stack. use a cuda 12.6+ image.")
PY

mkdir -p runs_v2 bundles_v2
echo "bootstrap ok"
