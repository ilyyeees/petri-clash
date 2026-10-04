# Regrowth lesson verification

Verified on 2026-10-04, on top of the seeded-duel, raster-efficiency and paired-comparison upgrades.

![Paused injury with exact cell counts and an explicit next action](lesson-injured.png)

## A guided experiment

The optional lesson exposes the shipped cultures' measured regeneration behavior. It starts a fresh 48×48 solo soft-growth field, guides a central seed, grows for exactly 160 ticks, previews a measured cut, pauses after applying it, and then offers slow observation or single stepping.

The cut is the smallest integer-radius disk at the rounded living-cell centroid that removes 25–60% of currently living cells. Centroid ties round to even, and disk membership uses the same inclusive squared-distance rule as the simulation. The preview reports its exact population loss. Every state channel is erased in that disk; the actual remaining count must match the preview before observation continues. An unavailable safe cut is reported explicitly.

The count goal is at least 90% of the pre-cut population, sustained for **24 consecutive simulation ticks**, within a maximum of 160 recovery ticks. Living means alpha > 0.1, matching the existing scoreboard. The hold was extended after visual review so the observation is less abrupt while color/opacity settles. Completion measures population count, not exact shape, color restoration, combat strength, or general model quality.

Only the visible culture's model advances. Calling an empty opponent would still consume RNG and change the trajectory. Growth and recovery stop exactly at each user-action gate and terminal result, including 8× or oversized step requests. Recovery plays at six simulation ticks per second at 1× while rendering retains the configured frame rate. Pause does not accumulate a catch-up burst.

## Measured shipped-model study

The shipping controller completed all **15 cases**: five ready cultures, simulation seeds 0/7/42, CPU, one Torch thread, 48×48 field. Every state remained finite, every preview matched the applied cut, and all terminal results stayed frozen.

| Culture | Selected radius | Recovery ticks at seeds 0 / 7 / 42 |
| --- | ---: | --- |
| Heart | 5 | 28 / 29 / 28 |
| Star | 4 | 28 / 27 / 28 |
| Sun | 4 | 27 / 27 / 28 |
| Flower | 7 | 28 / 27 / 27 |
| Umbrella | 4 | 28 / 27 / 27 |

See [portable counts, exact cut plans and runtime versions](lesson-results.json) and the [reproduction script](verify_lesson.py). These are bounded smoke measurements, not a statistical recovery guarantee or ranking. The models, weights and v1 were not changed or retrained.

## Input and state safety

- Enter/button equivalents avoid requiring precise mouse gestures. Shift-click plants at the seed marker; normal click applies the marked cut. Static outlines, crosshairs and instructions remain available with reduced motion.
- Repeated primary actions in one event batch cannot skip the paused injury display. Free damage, planting, clearing and rule changes stay locked; reset and culture changes begin a clean lesson.
- Exit restores the previous sandbox grid, rules and speed. Starting a duel leaves the lesson cleanly. Direct `--lesson` startup defers the unused opponent; an unavailable opponent at exit receives an explicit current-culture fallback notice rather than blocking the solo lesson or trapping its exit.
- A queued Shift press/click/release now uses event-order tracking instead of only polling an already-released key. Both Shift keys and focus loss are covered. This also improves ordinary sandbox planting.
- Non-finite inputs or proposals invalidate the observation. Failed model selection preserves the current state and CPU RNG. Reports distinguish complete, timeout, unavailable and invalid observations.
- Teaching checkpoints snap to the exact frozen board and clear old seed/cut effects, while live animation remains presentation-only.

## Regression and native-window checks

The full checkpoint-enabled suite passed **235 tests and 565 subtests**, alongside compilation and whitespace checks. New coverage includes every controller transition, exact fraction bounds, all 512 small 3×3 masks against an independent geometry reference, hold reset, end-boundary completion, timeout, invalid states, export isolation, bitwise replay at different batch sizes, all-channel damage, solo inference, repeated inputs, failed loads, paused pacing, and minimum/default/large layouts.

A native cloud-desktop walkthrough verified Shift-click planting, the exact tick-160 growth stop, ordinary crater click after Shift release, paused injury, N single-step, slow Space playback, live speed-label updates, terminal freeze, retry, culture restart, and exit. Heart at seed 42 showed 242 cells before, 164 after the cut, and 242 at completion after 28 recovery ticks. The native display remained around 30 FPS during that walkthrough; this is an observation on this host, not a display-latency benchmark or a cross-device performance guarantee.

Legacy hard/soft sandbox and duel trajectories, exact results, and Python/NumPy/Torch RNG states were also compared against the preceding revision with identical inputs. The earlier raster benchmark now defaults to its two measured historical revisions, so this lesson's intentionally new button does not invalidate that historical pixel-equivalence check.

CPU Linux execution was checked. GPU execution, Windows/macOS native windows and assistive-technology integration were not verified; keyboard alternatives alone do not establish accessibility compliance or screen-reader support.

## Reproduce

```bash
python v2/clash.py --device cpu --lesson
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests -q
python -m compileall -q v2
python v2/verification/verify_lesson.py --output /tmp/lesson-results.json
python v2/verification/capture_lesson.py --phase injured --output /tmp/lesson-injured.png
python v2/verification/capture_lesson.py --phase damage --width 860 --height 740 \
  --output /tmp/lesson-preview.png
```

Design sources: the original [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) demonstrates erase/regrow interaction; [Xbox objective-clarity guidance](https://learn.microsoft.com/en-us/xbox/accessibility/xbox-accessibility-guidelines/109) and [interactive tutorial guidance](https://gameaccessibilityguidelines.com/include-interactive-tutorials/) informed the optional, replayable stages and persistent instructions. The local measurements above validate these particular checkpoints within the tested scope.
