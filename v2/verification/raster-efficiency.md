# Raster efficiency verification

Verified on 2026-10-04 against `a31d6c84ae3cd3d77f7436e54b4d1e817f5f14f5`, the seeded-duel revision. This changes presentation work only; simulation, scoring, weights, and `v1/` are untouched.

## Two bounded changes

- Expire effects before allocating their board-sized alpha overlay. When none remain, return without allocation or blitting. Active effects retain the same rendering and reduced-motion behavior.
- Scale the nine fixed-size culture thumbnails to 28×28 once at UI creation. Drawing sets their normal/dim alpha explicitly, so availability changes cannot leave a ready culture dimmed. Window resizing does not change the picker size.

Stats refresh, animation, simulation timing, and input handling are unchanged. The patch introduces no board, score, or simulation-state cache.

## Actual-checkpoint measurements

The [reproducible benchmark](benchmark_raster_ui.py) uses shipped heart `seed_000` and star `seed_001`, a 48×48 board, seed 42, CPU-only PyTorch, and one Torch thread. It compares both revisions at default 1100×902 and minimum 860×740 window sizes, at 1× and 8× speed. Four repeats alternate version, size, and speed order, with 30 timed frames per trial after 20 simulation warmup ticks and 10 draw warmup frames. Each microbenchmark has 300 samples per trial. Loading and warmup are excluded.

Whole-draw latency, milliseconds:

| Window | Speed | Before p50 / p95 | After p50 / p95 |
|---|---:|---:|---:|
| 1100×902 | 1× | 3.521 / 4.380 | 2.866 / 3.278 |
| 1100×902 | 8× | 3.542 / 3.846 | 3.106 / 3.584 |
| 860×740 | 1× | 2.620 / 3.032 | 2.415 / 3.072 |
| 860×740 | 8× | 2.893 / 3.267 | 2.654 / 3.197 |

The empty-effects microbenchmark falls from about **0.366 ms** median at the default size and **0.177 ms** at minimum size to **under 0.001 ms**. Sidebar median cost falls by about **0.033–0.039 ms**. These isolated costs support the two optimizations; whole-frame differences also include host scheduling noise.

The benchmark verifies identical final screen pixels, all four simulation tensors, and Python/NumPy/Torch RNG states in every matched run. It exports equality booleans rather than raw fingerprints. [Full measurements, software versions, and per-trial summaries](raster-efficiency.json).

## Regression coverage

The full CPU suite with shipped checkpoints enabled passed **147 tests and 180 subtests**. The new focused tests check:

- Empty effects and the exact 1.25-second expiry boundary allocate no overlay and leave pixels unchanged
- Live damage/seed effects still draw; reduced motion suppresses the moving rays and radius growth
- Effects expire while paused without advancing simulation ticks
- All nine thumbnails are scaled only once and match uncached alpha compositing at both sizes, including unavailable → ready → unavailable transitions
- Drawing preserves all four simulation tensors and all three random generators

Existing feedback and duel tests continue to cover paused edits, view changes, resizing, repeated actions, replay, result controls, and exact scoring.

## Reproduce

From the repository root with CPU dependencies and the baseline Git revision available:

```bash
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests -q
python -m compileall -q v2
python v2/verification/benchmark_raster_ui.py --output raster-efficiency.json
```

SDL dummy measures raster work, not native display/compositor behavior, vsync, input-to-display latency, or FPS pacing. Timed scenes have no active action effects. Median draw cost improved in all four configurations, but tail latency is noisy and did not improve in every case. Simulation dominates 8× play, and total unpaced-frame time does not consistently improve; this is not a claimed simulation or universal gameplay-FPS speedup.
