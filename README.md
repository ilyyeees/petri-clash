# Petri Clash

Two neural cellular automata grow, collide, and regenerate on a shared grid.

- **[v2: Living arena](v2/README.md)**: the actively improved version, with mouse-only lab tools, guided regrowth, saveable seeded duels, paired comparisons, territory/pressure views, and CPU-first play. Includes trained weights; no GPU or training is needed to try it.
- **[v1](v1/)**: the original, self-contained baseline. Unchanged by the v2 upgrade.

Quick start (Python 3.11+):

On Apple silicon, use `python -m pip install torch` instead of the CPU-index
command below. For other platforms, the [official PyTorch installer](https://pytorch.org/get-started/locally/)
can select a compatible wheel.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r v2/requirements.txt
python v2/clash.py --device cpu
```

The two versions do not share code, targets, or weights. See the [verification reports](v2/verification/README.md) for measured results and limitations.
