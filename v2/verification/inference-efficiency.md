# CPU inference efficiency

Verified on 2026-10-04 against the original NCA source at
`c5209c773f3283556c24ba49248f6b5324cd7bc6`. The candidate is the guarded production
`NCA.step` in this revision, not a separate prototype. No weight, battle rule,
scoring, RNG, precision, or tensor-layout change is made.

## Small change and guarded boundary

The ordinary CPU inference path applies ReLU in place to `fc0`'s fresh temporary
activation before passing it to `fc1`. At a 48×48 grid, 256 hidden channels and
float32, this avoids one **2.25 MiB output allocation per model proposal**. This
is an allocation-size calculation, not a measured reduction in process RSS.

The fast path requires eval mode, CPU, eager execution, inference mode, an exact
`nn.Conv2d` first layer with no instance-level forward override, and no local or
global forward/pre-hooks. Training, ordinary autograd, `no_grad` alone, GPU,
compiled execution, custom layers and instrumented layers use the original
out-of-place ReLU. The gate is checked before `fc0` executes, including hooks
that could retain or replace its activation and then remove themselves.

The temporary activation is reused; the organism input, model parameters and
buffers are not mutated. Checkpoint shapes and the public model API are unchanged.

Primary API references: PyTorch documents the optional [in-place ReLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html),
[inference mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html)
as a thread-local autograd context distinct from eval mode, and
[`torch.compiler.is_compiling`](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.is_compiling.html)
as the compiler-capture predicate. The guards and equivalence audit here apply
those APIs to this model; the measured speedup comes from this benchmark.

## Measured single-tick CPU latency

Each value is pooled p50 / p95 in milliseconds from four alternating-order
baseline/current pairs, 240 timed ticks per trial after 40 warmup ticks. The
[reproducible harness](benchmark_inference.py) loads the baseline NCA source from
Git and temporarily selects its exact `step` method or the current production
method. No source files or checkpoints are modified by the harness.

| Cultures / rules | Before p50 / p95 | Current p50 / p95 | Lower p50 |
|---|---:|---:|---:|
| heart-star / hard | 7.257 / 8.497 | 5.527 / 6.484 | 23.8% |
| heart-star / soft | 5.772 / 7.409 | 3.917 / 4.656 | 32.1% |
| heart-flower / hard | 7.742 / 10.045 | 5.509 / 6.754 | 28.8% |
| heart-flower / soft | 5.640 / 6.743 | 3.944 / 5.028 | 30.1% |

Timed scenes are ordinary sandbox runs with seeded-random positions, not scored
duels. Heart/star uses simulation seed 0 and starts (14,26)/(32,26); heart/flower
uses seed 7 and starts (10,25)/(38,25). Normal `Arena.step` validation and hard-mode
battle rules are included. Loading, warmup, rendering, event handling and FPS
pacing are excluded. Each matched timing pair also checks final state bits,
dtype/strides and Python/NumPy/Torch RNG equality.

Both variants retain one Torch intra-op thread and consume about one CPU-second
per wall-second during stepping. The faster path reduces total CPU time for a
fixed number of ticks; it does not introduce simulation worker threads. The
[full report](inference-efficiency.json) retains all samples, per-trial summaries,
pooled summaries, CPU seconds, versions, culture/checkpoint seed identifiers,
placements and invariant results without absolute checkpoint paths or fingerprints.

## Eight-tick frame work

These measure `Arena.step(8)` only, with four alternating-order pairs and 60 timed
batches per trial after 40 warmup ticks. They do not measure a rendered frame.

| Cultures / rules | Before p50 / p95 | Current p50 / p95 | Lower p50 |
|---|---:|---:|---:|
| heart-star / hard | 61.443 / 74.089 | 43.427 / 47.479 | 29.3% |
| heart-star / soft | 46.227 / 58.175 | 32.236 / 36.082 | 30.3% |

The default UI targets 30 FPS, or 33.3 ms per frame including drawing and events.
Hard-mode eight-tick work still exceeds that budget. Soft-mode median simulation
work is near the budget before rendering, and its tail can exceed it. This is a
simulation-efficiency improvement, not a claim of guaranteed 30 FPS at 8× speed.
No tick budgeting, frame pacing, speed semantics or duel scoring was changed.

## Exact full-round audit

Separate, untimed duels use mirrored positions (16,23)/(31,23), 600 total ticks
and 60 warmup ticks. At every tick the production and original paths match:

- Exactly one tick advances on each side; both arena and duel clocks match the expected tick
- All bytes of both organism states, ownership and control, including signed zero
- Dtype and strides
- Python, NumPy and Torch RNG state
- The complete duel snapshot, including exact cell-tick totals and winner
- The candidate's previous input tensors remain unchanged

Every parameter and registered buffer in both runs is also bitwise unchanged.
All four rounds are required to finish valid at exactly tick 600. The full audit
was rerun after adding explicit per-iteration progress and terminal checks, so a
matching early-invalid round cannot be counted as a complete replay. The close
heart/flower hard result remains exact.

| Cultures / rules | Simulation seed | Left / right cell-ticks | Winner |
|---|---:|---:|---|
| heart-star / hard | 0 | 129,056 / 91,028 | left |
| heart-star / soft | 0 | 130,659 / 98,301 | left |
| heart-flower / hard | 7 | 123,178 / 121,494 | left |
| heart-flower / soft | 7 | 130,619 / 132,816 | right |

Guard, retained-activation hook, custom-layer, input-immutability and autograd
regressions are covered by [the focused tests](../tests/test_inference_efficiency.py).
The full CPU suite on the final guarded code passed **249 tests and 616 subtests,
with no skips**. Compilation, diff checks, eight-tick UI, lesson UI, bounded duel,
paired-comparison and standard benchmark smokes also passed. The benchmark
refuses Python `-O`, because its invariant assertions must execute.

## Controls, alternatives and limits

This run used Linux x86-64, Python 3.12.14, Torch
2.14.1+cpu, NumPy 2.3.5, float32 NCHW
states, 24 channels, 256 hidden units, and fire rate 0.5. The shipped models were
heart `seed_000`, star `seed_001`, and flower `seed_002`. CPU affinity exposed nine
logical CPUs; the execution sandbox did not expose a host process inventory or
cgroup CPU quota. The known Petri Clash native QA window was closed normally
before measurements, and other task-owned CPU-heavy tests were held during the
final timing run. Unrelated host activity cannot be ruled out.

Deterministic algorithms and deterministic uninitialized-memory filling stayed
enabled. The latter can amplify the cost of allocating temporary tensors; the
measured gains should not be generalized to other PyTorch builds, hardware,
threading choices or deterministic settings. No process-RSS reduction is claimed.
GPU paths were preserved but not run, and cross-device/version bitwise replay
is not promised.

Exploratory profiling identified convolution, activation and allocation work as
the main costs; entering/exiting inference mode alone was about one microsecond.
Testing 1/2/4/9 intra-op threads did not justify changing the one-thread default:
more threads increased resource use, and higher-thread tails were worse in key
cases. A separate persistent-worker proposal prototype was faster on this
multicore host, with exact seeded replays, but used about 1.5–1.7 active cores and
added RNG/fallback/lifecycle complexity. Under forced one-core affinity its gains
fell to single digits. That prototype is deliberately not shipped. The smaller
serial activation reuse is the chosen change.

## Reproduce

From the repository root with CPU dependencies, shipped checkpoints and the
baseline Git revision available:

```bash
python v2/verification/benchmark_inference.py --output inference-efficiency.json
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests/test_inference_efficiency.py -q
```

Defaults reproduce the measurement/audit sizes above and take a few minutes on
the measured host. A small functional smoke is available without waiting for
full performance sampling:

```bash
python v2/verification/benchmark_inference.py --steps 2 --frame-steps 8 \
  --warmup 0 --repeats 1 --replay-ticks 2 --replay-warmup 0 --output smoke.json
```

A short smoke verifies the workflow, not the documented full-round or performance
claims. `--baseline-ref` can select another locally available source revision;
the default pins the measured original. The candidate always uses the current
checkout, so future simulation changes may appropriately fail equivalence.
