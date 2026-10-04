# Declared dependency floor

Verified on 2026-10-04 after the evaluation-metadata type fix. The declared lower
bounds were installed together in a fresh isolated environment rather than
inferred from the current development environment.

| Component | Tested lower-bound version |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.6.0+cpu |
| NumPy | 1.24.0 |
| Pillow | 10.0.0 |
| Pygame | 2.5.0 |
| TensorBoard | 2.14.0 |

The exact runtime combination resolved successfully using the official PyTorch
CPU index and PyPI. Development-only pytest was allowed to use its current
compatible release. Application, trainer and TensorBoard imports, dependency
health and compilation passed. The repository, existing environments, bundled
weights and `v1/` were not modified by this compatibility check.

## Results

- **780 tests and 1,121 subtests passed**, with shipped checkpoint checks enabled;
  no failures, errors or skips
- **236 focused checkpoint tests passed**, including all 23 bundled checkpoint
  files and the historical NumPy RNG compatibility path
- Hard and soft headless runs passed with no usable SDL driver
- Dummy-SDL hard/soft arena and guided-lesson window smokes passed
- TensorBoard logging and readback passed
- A fresh duel export/rerun matched **7,079 / 4,947** cell-ticks
- The published 600-tick heart/flower recipe matched **123,178 / 121,494**, with
  the expected Python, PyTorch and NumPy version warnings

The same final code also passed all **780 tests and 1,121 subtests** on the
Python 3.12.14 / PyTorch 2.14.1+cpu / NumPy 2.3.5 environment and the fresh
Python 3.13.5 / PyTorch 2.14.1+cpu / NumPy 2.5.3 environment. The latter emits an
existing Pygame `pkg_resources` deprecation warning.

No compatibility-driven change to the declared minimum requirements was needed.
These results establish the tested Linux x86_64 CPU combinations. They do not
cover every permitted dependency combination or Python patch, native display,
macOS/Windows, Conda resolution or accelerators. Matching these example scores
does not establish general cross-version bitwise determinism.

## Reproduce the floor check

For normal installation, follow the [main quick start](../README.md) and the
[official PyTorch guidance](https://pytorch.org/get-started/locally/). The pinned
versions below document this isolated compatibility check:

```bash
python3.11 -m venv .venv-minimum
source .venv-minimum/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.24.0 pillow==10.0.0 pygame==2.5.0 \
  tensorboard==2.14.0 'pytest>=8'
python -m pip check
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests -q
python -m compileall -q v2
python v2/replay.py v2/verification/duel-recipe-example.json
```

Use a separate environment. This check requires no GPU, checkpoint replacement
or training sweep.
