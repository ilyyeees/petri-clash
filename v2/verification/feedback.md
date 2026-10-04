# Arena feedback and tactical HUD verification

Follow-on to `ab6e2bd061ee704147bbf9e628f4eb047e28ea7b`, verified on 2026-10-04. The previous living-arena upgrade is included in this revision. No model weights, simulation rules, or v1 files changed in this follow-on.

![Tactical arena with real paused damage feedback](arena-feedback.png)

## What changed

- A dominant 740px field at the default 1100×902 window, a fixed 16px gutter, and a compact command rail replace the wide score card and dead space. Warm-neutral chrome, flat beveled controls, stronger headings and tabular numbers follow the reference research linked in the [main guide](../README.md#interface-direction).
- Exact held-cell counters in hard mode, living-cell comparisons in soft mode, explicit lead margins and full-board percentages. Hard-mode bars include neutral land. Soft-mode bars are independent because cultures can overlap. No final winner is declared.
- Recent net changes use the actual observed simulation-tick span, normally 30 ticks at 1× and 32 at 8×, and never infer combat kills from shrinking tissue.
- Board-only time-based easing; view/mode/grid/color changes and edits snap correctly. No simulation state or RNG changes are used for animation.
- Damage pulses show exact living cells erased on each side. Planting uses side-colored markers. Paused actions update scores immediately, effects expire while paused, and replay/clear/culture changes discard old effects and trends.
- Reduced-motion mode bypasses easing and moving effects; F or the FX button toggles it.

## Regression and visual coverage

The final full run passed **96 tests and 46 subtests**, with `PETRI_TEST_CHECKPOINTS=1` and no skips. `compileall`, `git diff --check`, and hard/soft headless smoke runs passed.

Independent review verified 1100×902, 860×740, 1800×740 and 860×1200 layouts; labels and controls remain inside their intended regions. The score typography also handles five-digit counts without overlap. Real shipped heart/star checkpoints were used for screenshot and layout inspection. Fast integration assertions additionally use clearly labeled small synthetic models.

Tests cover all Python/NumPy/Torch RNG states and all simulation tensors remaining unchanged by drawing, replay with and without drawing/effects, paused planting/damage, exact scores before the next draw, mode/view/culture changes, repeated inputs, keyboard/mouse controls, resizing with fresh hit regions, bounded effect history and expiry. Both original and refined UI trajectories have identical organism, ownership and control hashes.

## Measured presentation overhead

Actual shipped heart/star models, 48×48 grid, one Torch thread, same 1100×902 window. Three alternating-order repeats per version/speed, 20 simulation warmup ticks, 10 draw warmup frames and 60 measured frames per repeat. Both versions refresh exact statistics every frame.

| Speed | Previous draw p50 | Refined draw p50 | Added raster cost |
| --- | ---: | ---: | ---: |
| 1× | 2.497 ms | 3.390 ms | 0.893 ms |
| 8× | 2.571 ms | 3.571 ms | 1.000 ms |

These are SDL-dummy raster drawing and flip timings, not native display latency. They exclude compositor, vsync, input handling and FPS pacing; no action effects are active in this timing run. The larger refined board and score/easing work are included. Host timings vary. No simulation speedup is claimed. See [complete UI measurements](feedback-performance.json), [matched simulation-only measurements](feedback-simulation.json), and the [reproducible UI benchmark](benchmark_feedback_ui.py).

## Reproduce

```bash
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests -q
python -m compileall -q v2
python v2/verification/capture_feedback.py --output /tmp/arena-feedback.png
python v2/verification/benchmark_feedback_ui.py --output /tmp/feedback-performance.json
```

The benchmark requires the fixed baseline commit in Git history. CPU execution was verified; GPU execution and real-monitor display latency were not measured.

## Current overview

The README now shows the same damage-feedback scene with the current cumulative
controls: lesson/duel entry, separate simulation/checkpoint labels, loaded seeds,
and mouse-only lab tools. The original capture above remains historical evidence.

![Current lab overview](arena-overview.png)

```bash
python v2/verification/capture_feedback.py --output arena-overview.png
```

This capture uses the shipped heart/star models and dummy SDL. It is a rendered
interface check, not a native-window interaction test.
