# CPU battle mask efficiency

Verified on 2026-10-04 against main commit
`cc2084df711ca07902b5a95552fa95354b10992b`, which already includes the prior NCA
activation and rendering optimizations. This is an additional hard-mode CPU
improvement. No NCA math, precision, weights, rules, scoring, seed policy, thread
policy, training code, UI behavior or `v1/` files were changed.

## Measured result

Four alternating-order baseline/candidate pairs per case, 240 timed single ticks
per trial after 40 warmup ticks. Eight-tick trials use 480 timed ticks per trial
(60 calls). Values below are pooled **p50 / p95 milliseconds per call**.
Normal `Arena.step` bookkeeping is included; loading, warmup, rendering, event
handling, frame pacing, profiling and correctness/resource audits are excluded.

| Hard-mode workload | Before p50 / p95 | Current p50 / p95 | Lower p50 |
|---|---:|---:|---:|
| heart/star, 32×32, one tick | 2.829 / 3.327 | 2.561 / 3.000 | 9.5% |
| heart/star, 48×48, one tick | 5.415 / 6.754 | 4.684 / 5.452 | 13.5% |
| heart/star, 64×64, one tick | 9.439 / 11.105 | 8.367 / 9.845 | 11.4% |
| heart/star, 96×96, one tick | 21.688 / 24.403 | 18.829 / 21.009 | 13.2% |
| heart/flower, 48×48, one tick | 5.302 / 5.762 | 4.660 / 5.295 | 12.1% |
| heart/star, 48×48, eight ticks | 43.695 / 49.394 | 38.028 / 42.162 | 13.0% |

The one-tick reductions are **9.5–13.5%** in this run. The default 48×48
heart/star workload falls from 5.42 to 4.68 ms. Two negative controls use the same
sampling/order: hard mode with the baseline on both sides differs by 0.7% in p50,
and the unchanged soft-mode path differs by 0.4%. Both variants consume about one
CPU-second per wall-second during stepping; no extra workers or CPU threads are
introduced.

The eight-tick median is still above a 30 FPS frame's 33.3 ms budget before
drawing. These are CPU simulation timings, not measured native-display FPS or a
guarantee of smooth 8× play. An earlier exploratory run had a worse p95 in one
case, illustrating why tail latency should not be treated as a hardware-independent
promise. The table above is the final guarded production run.

The [raw report](battle-mask-efficiency.json) retains every timing sample, per-trial
summaries, runtime versions, model identifiers, source checksum, negative controls,
replays, selected allocation-profile operators and resource samples. It omits
host paths, checkpoint fingerprints and final-state fingerprints.

## Smaller masking pipeline

Profiling directed this work toward simulation filtering rather than another UI
cache. The production change is deliberately bounded:

1. **Reduce the CPU finite-cell predicate before testing it.** For a nonempty
   real-floating channel vector, `abs().amax()` is finite exactly when every
   channel is finite: absolute value preserves finiteness, either infinity becomes
   positive infinity, NaNs propagate, and a comparison reduction cannot overflow
   by summing. Only this boolean decision changes implementation; surviving state
   values are copied unchanged. Supported CPU float16, bfloat16, float32 and
   float64 use it in eager execution. Empty channels, other dtypes, compilers and
   other devices retain the original predicate.
2. **Combine cell masks before broadcasting across channels.** Finite/territory
   and finite/support filters each need one full-state output instead of two.
   Life is calculated from the small alpha/territory mask before one final
   full-state filter. Strict checks still inspect all raw input/proposal cells,
   including cells subsequently excluded by territory or support.
3. **Use scalar ownership labels.** Hysteresis and abandoned-land cleanup no
   longer allocate full owner-sized zero/one/two fill tensors.

Fusion is restricted to positive, four-dimensional tensors with the exact
canonical dense CPU strides, outside compiler capture. Original nested masking
is retained elsewhere. Merely checking `is_contiguous()` is insufficient because
it ignores singleton-axis strides. Review found and fixed both a singleton
output-stride change and an expanded-input/permuted-mask change that altered
`rand_like` value placement even while its final RNG state matched. The fallback
preserves those observable layouts rather than relaxing the invariance checks.

At 48×48, 24 channels, float32, the common hard path eliminates six full-state
`where` outputs per tick: **1.266 MiB of temporary output allocations** by shape
calculation. The focused dispatch regression checks twelve full-state passes in
the original/compiler path versus six in the optimized path. The instrumented
profile also records the added `abs`/`amax` work and smaller boolean reductions.
These allocation figures are not a measurement of peak live memory or a promise
of lower process RSS. No persistent scratch storage or new cache is introduced.

## Exactness and compatibility

No numerical tolerance was introduced:

- Four separate 600-tick scored duels finish valid at the exact endpoint:
  heart/star and heart/flower on 48×48, plus heart/star on 32×32 and 64×64.
- At every tick, both state tensors, ownership and control match **every byte,
  dtype and stride**, along with Python/NumPy/Torch RNG, full duel snapshots and
  exact cell-tick scores. Inputs, model parameters and buffers remain unchanged.
- Board RGB is checked in life/territory/pressure views with both color settings
  at every 100-tick boundary: 144 exact image-array comparisons across the four
  replays. Rendering code itself is unchanged.
- The full 48×48 scores remain 129,056 / 91,028 for heart/star (seed 0), and
  123,178 / 121,494 for heart/flower (seed 7).
- Finite-cell tests cover all 65,536 bit patterns of float16 and bfloat16 in each
  of eight test channels, random full float32/float64 bit patterns, NaN/infinity mixtures,
  maximum finite values, signed zero and actual subnormals. All checks preserve
  input bytes and strict failure behavior.
- Committed tests cover independently varied input/proposal layouts, observer
  retention, empty/singleton dimensions, expanded views, permuted masks and
  compiler/non-CPU predicate dispatch. An additional independent matrix passed
  9,072 tiny mixed-dtype/layout/observer/RNG cases before final measurement.

Full suites on the final production code each passed **797 tests and 1,789
subtests, with no skips**:

| Python | Torch CPU | NumPy | Pillow | Pygame |
|---|---|---|---|---|
| 3.11.16 | 2.6.0 | 1.24.0 | 10.0.0 | 2.5.0 |
| 3.12.14 | 2.14.1 | 2.3.5 | 12.3.0 | 2.6.1 |
| 3.13.5 | 2.14.1 | 2.5.3 | 12.3.0 | 2.6.1 |

All three stacks also passed syntax compilation, a hard headless run, bounded
duel export/replay with matching saved scores, paired comparison, lesson startup,
eight-tick dummy-SDL UI smoke with an exit snapshot, and the standard benchmark
smoke. `git diff --check` passed. The first row verifies the declared runtime
minimums; no dependency increase is needed. Python 3.13 reports the existing
Pygame `pkg_resources` deprecation warning; it does not fail a test.

## Bounded resource check

After timings/profiling, the final implementation advanced **5,000 hard-mode
ticks** on a 48×48 seeded heart/star sandbox. Every applied state stayed finite;
all 20,000 retired state/owner/control objects were reclaimable, and the model
cache stayed at two entries. After explicit GC every 500 ticks, sampled resident
memory ranged from **496.69 to 497.27 MiB** and was unchanged from
tick 1,000 through 5,000. Both cultures remained living at every sample.

This process had already loaded larger grids and run allocation profiling, so
its RSS is not a cold-start or standalone-game measurement. This is a bounded
retained-resource observation, not a peak-memory comparison or proof that every
workload is leak-free. GC and diagnostic samples are excluded from timings.

## Reproduce and limits

Run from the Git checkout with shipped checkpoints and CPU development dependencies:

```bash
python v2/verification/benchmark_battle_masks.py --output battle-mask-efficiency.json
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests -q
```

The harness pins the original battle source from Git and swaps only the battle
function while using identical current NCA/runtime code. It rejects Python `-O`,
checks the candidate source stayed unchanged during measurement, and restores the
selected function on exit. A small workflow smoke, without performance claims:

```bash
python v2/verification/benchmark_battle_masks.py --steps 2 --frame-steps 8 \
  --warmup 0 --repeats 1 --replay-ticks 2 --replay-warmup 0 \
  --resource-ticks 2 --output smoke.json
```

The measured stack was Linux x86-64, Python 3.12.14, Torch 2.14.1+cpu, NumPy 2.3.5,
float32, one intra-op thread and nine visible affinity CPUs. Deterministic
algorithms and uninitialized-memory filling stayed enabled. Other task-owned
CPU-heavy jobs were held during timing; unrelated host activity and CPU quota
were not measured. Accelerators, native-display FPS, long training and
cross-version/cross-device bitwise replay are not claimed. Non-CPU predicate and
fusion dispatch use the original paths; shared scalar ownership fills were not
executed on an accelerator.

Alternative NCA delta/final-output reuse was discarded for marginal gains, an
alternate matrix projection was discarded after an exactness counterexample,
and `aminmax` was discarded for inconsistent performance. Profiling did not
justify another font cache or changing duel bookkeeping. This revision retains
only the measured battle-pipeline change and its compatibility fallbacks.
