# Frozen-board rendering

Verified on 2026-10-04. This is a small presentation-only optimization: one scaled
field image can be reused on a paused frame when its final RGB pixels and output
size are unchanged. NCA inference, match rules, scoring, model loading, weights
and `v1/` are unchanged.

## What is reused

The key is the **actual post-blend uint8 RGB image plus the field's pixel size**.
This matters because a paused organism may still be easing toward a newly edited
image. A tick count, model identity or pause flag alone would not be sufficient.
The cache owns a copy of the small RGB key and retains at most one scaled Surface.

Changed pixels or field size rebuild the raster from the original RGB image.
Pointer cues, damage/planting effects, lesson markers, field labels, results and
other controls continue drawing on the window; they never paint into the retained
field Surface. Paused edits, STEP, view/color/FX switches and resizing therefore
keep exactly the same visible result.

Live frames clear the retained image and Surface and take the original raster
path, without a cache comparison or key copy. Existing controller refresh already
pauses completed duels and lessons, so no new lifecycle state or invalidation
counter is introduced. Explicit exit snapshots retain their final-state redraw.

This follows the semantics of [Pygame's scaling operation](https://www.pygame.org/docs/ref/transform.html#pygame.transform.scale):
a cached result is reusable only for the same source pixels and destination size.
The cache uses [exact array equality](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html),
not approximate color matching.

## Measured cost

Shipped heart seed 000 and star seed 001, 48×48 hard mode, one CPU thread,
Python 3.12.14 / PyTorch 2.14.1+cpu / NumPy 2.3.5 / Pygame 2.6.1. Three repeats of
120 measured frames per scene, after 160 growth ticks and 20 warmup frames.

Each current frame measures the original field-raster path, an unchanged copy of
that path as a control, and the candidate. Their order rotates every frame.
Live scenes advance one simulation tick before the comparisons; presentation
easing advances once outside timing. All three timed draws then see the same
post-blend image. Pixel/state/RNG/hitbox checks occur outside the timed interval.

| Scene | Original median / p95 | Cached median / p95 | Median paired cache delta | Unchanged-control delta |
|---|---:|---:|---:|---:|
| 860×740, settled paused | 2.618 / 3.383 ms | 2.366 / 3.080 ms | −0.237 ms | −0.027 ms |
| 1100×902, settled paused | 3.067 / 3.506 ms | 2.580 / 3.057 ms | −0.492 ms | +0.001 ms |
| 860×740, live | 2.580 / 3.011 ms | 2.596 / 3.051 ms | +0.006 ms | +0.003 ms |
| 1100×902, live | 3.130 / 3.710 ms | 3.128 / 3.738 ms | +0.013 ms | +0.019 ms |

Warm, settled paused-lab median drawing cost fell by about **10% at minimum size**
and **16% at default size**. Live differences are small relative to the unchanged
control; no live speedup is claimed. The timed pre-draw warms the paused cache,
so these measurements do not establish first-pause, edit, easing-miss or resize
cost. Frozen duel/lesson correctness is tested separately; their latency was not
measured by this table.

The retained Surface pitch plus RGB storage was about **1.03 MiB / 2.10 MiB** at
the two sizes and zero during live frames. This is not a process-RSS measurement
and excludes object overhead and transient allocations. Loading, simulation,
frame pacing and native display/compositing are excluded from the timing.
Competing host work is not controlled, and these results are not a universal FPS
claim. [Full measurement report](frozen-board-rendering.json).

General text caching was also profiled and rejected: it saved little full-frame
time, gave inconsistent live gains, and required more cache-key/lifetime logic.

## Verification and fresh interpreter check

The complete opt-in CPU suite passed **716 tests and 1,121 subtests** on both:

| Runtime | Python | PyTorch | NumPy | Pillow | Pygame | pytest |
|---|---|---|---|---|---|---|
| Existing test environment | 3.12.14 | 2.14.1+cpu | 2.3.5 | 12.3.0 | 2.6.1 | 9.1.1 |
| Fresh isolated environment | 3.13.5 | 2.14.1+cpu | 2.5.3 | 12.3.0 | 2.6.1 | 9.1.1 |

No skips, failures or errors occurred. Compilation, dependency health and
whitespace checks passed. The Python 3.13 run emitted Pygame's existing
`pkg_resources` deprecation warning; it did not affect these checks.

The 88 new cache cases compare full-window pixels and hitboxes with an uncached
reference across pause/resume, easing, edits, culture selection, resizing,
views/colors/FX, every lesson phase, finished/hidden duel results and queued-action
exit snapshots. They also check tensor/RNG/controller stability, input ownership,
unchanged overlay-source pixels, one-entry retention, and no live comparisons or
copies. A separate bundled-model pass matched all 186 transition snapshots.

Eight fresh Python 3.13 CLI smokes passed: headless hard/soft play without a usable
SDL driver, dummy-SDL hard/soft/lesson windows, and portable export/rerun. A fresh
recipe matched **1,784 / 1,049** cell-ticks. The published 600-tick heart/flower
example still matched **123,178 / 121,494**, while correctly warning about the
Python and NumPy version changes. That is evidence for these runs, not a general
cross-version bitwise guarantee.

All checks here used Linux CPU and dummy SDL. Native-display behavior, other
operating systems and accelerators were not tested. No training or checkpoint
export was needed for this optimization.

## Reproduce

```bash
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests -q
python -m compileall -q v2
python v2/verification/benchmark_frozen_board.py --frames 120 --warmup 20 \
  --repeats 3 --output frozen-board-rendering.json
python v2/replay.py v2/verification/duel-recipe-example.json
```

The benchmark embeds only the original uncached raster operations as its
reference; it does not require a historical Git checkout. It refuses Python
`-O` so equality assertions cannot silently be disabled. It exports runtime,
configuration, aggregate timings and equality outcomes without local checkpoint
paths, raw state/RNG values or fingerprints.
