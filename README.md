# Petri Clash

Two neural cellular automata grow, collide, and regenerate on a shared grid.

- **[v2: Living arena](v2/README.md)**: the actively improved version, with a visual culture picker, territory/pressure views, portable checkpoints, and CPU-first play. Includes trained weights; no GPU or training is needed to try it.
- **[v1](v1/)**: the original, self-contained baseline. Unchanged by the v2 upgrade.

Quick start (Python 3.11+):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r v2/requirements.txt
python v2/clash.py --device cpu
```

The two versions do not share code, targets, or weights. See the [CPU verification report](v2/verification/README.md) for measured results and limitations.
