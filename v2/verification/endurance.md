# Bounded endurance and resource stability

Checked on 2026-10-04 against `d1637bb`, after the full regression and clean
source-overlay installation checks. Production code, rules and bundled weights
were not changed by these diagnostics. Both checks used the shipped CPU models
on Linux, Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5 and
Pygame 2.6.1.

These are bounded stability checks, not proof that every workload is leak-free.
Elapsed time includes instrumentation and is not a performance benchmark.

## Long sandbox runs

Four fresh processes ran heart 000 versus star 001: hard and soft modes, each
repeated once, seed 42, 48×48, one intra-op CPU thread. Each process advanced
**10,000 ticks**, for **40,000 ticks total**. The normal sandbox rules and update
path were used without changing model inference or adding model hooks.

- Applied organism states and control remained finite at **every tick**
- At each 500-tick checkpoint, state bytes/layouts and Python, NumPy
  and PyTorch CPU RNG states matched the corresponding fresh-process repeat
  exactly: **20 comparisons per mode**
- Each run kept exactly two cached models
- Both cultures had living cells at every 500-tick sample; observed minima were
  218 heart cells and 181 star cells, not a claim about every intermediate count
- Final living counts were 219/182 in hard mode and 219/183 in soft mode

Current resident memory was sampled from Linux `/proc/self/statm` after explicit
`gc.collect()` every 500 ticks. The table contains sampled resident memory, not
an instantaneous peak or a historical high-water mark.

| Mode / repeat | First sample (MiB) | Final sample (MiB) | Sampled range (MiB) |
|---|---:|---:|---:|
| Hard / 1 | 368.02 | 367.94 | 367.30–368.15 |
| Hard / 2 | 365.49 | 366.48 | 365.49–366.48 |
| Soft / 1 | 364.45 | 364.80 | 363.10–365.63 |
| Soft / 2 | 364.23 | 365.21 | 363.31–365.63 |

Traced Python current memory grew by about 16–20 KiB per run, including the
harness's retained scalar samples and private comparison strings. Native tensor
storage is not comprehensively tracked by Python `tracemalloc`. The explicit GC
means this measures retained memory at those boundaries rather than normal
collection scheduling. Hard-mode raw proposals were not separately instrumented;
the finite checks cover the applied states and control.

Run the same sandbox workloads from the repository root with:

```bash
python v2/clash.py --device cpu --cpu-threads 1 --seed 42 --mode hard \
  --grid-size 48 --left 1 --right 2 --headless-frames 10000
python v2/clash.py --device cpu --cpu-threads 1 --seed 42 --mode soft \
  --grid-size 48 --left 1 --right 2 --headless-frames 10000
```

These commands reproduce the simulation workload; they do not add the diagnostic
per-tick assertions, private intermediate comparisons or RSS sampling. Repeating
on different software or hardware is not a general bitwise-determinism promise.

## Repeated UI lifecycle

Two fresh processes each completed **120 cycles, 15,021 simulation ticks and
4,399 draws** using dummy SDL, seed 1729 and one intra-op plus one inter-op CPU
thread. Left cultures cycled heart → star → sun → flower → umbrella; right
cultures used sun → flower → umbrella → heart → star. Hard and soft alternated.

Each cycle exercised sandbox reset/clear/plant/damage, 16 sandbox ticks, two
48-tick duels with eight warmup ticks, result hide/show, same-seed rematch,
lesson entry/exit, paused/live rendering, and resizing to 860×740 and 1100×902.
Six lessons per process ran through completion (cycles 1, 22, 43, 64, 85 and
106); the other 114 exited after four growth ticks. Every cycle ended at a fresh,
paused hard sandbox. Draw time was supplied explicitly; there was no FPS pacing.

Across both runs:

- **480 completed duels** and **240 exact within-process rematches**
- **12 completed lessons** and **228 clean early lesson exits**
- All **120 cross-process state/RNG/UI boundary comparisons** and 120 duel-outcome
  comparisons matched exactly
- 1,200 paused-raster reuse checks, 1,200 raster-invalidation checks, 1,200 frozen
  draw state/RNG checks and 480 frozen-resize checks passed
- No application exception or measured-run assertion failed

RSS was sampled before and after GC at identical end-of-cycle boundaries.
Excluding the first 20 warmup cycles leaves **100 samples per process**. The
before/after boundary values had no measurable GC reduction in these runs.

| Repeat | First post-warmup (MiB) | Final (MiB) | Sampled peak (MiB) | Post-warmup range width (MiB) |
|---|---:|---:|---:|---:|
| 1 | 382.61 | 381.98 | 383.99 | 3.25 |
| 2 | 382.63 | 382.76 | 384.10 | 1.46 |

The last ten post-warmup sample means differed from the first ten by −0.270 MiB
and +0.113 MiB. Startup was about 347.8 MiB before the full workload populated
its caches. These observations show no sustained large growth in this schedule;
they are not a universal memory ceiling or a leak-free guarantee.

Across 8,768 container observations per process, the model cache reached five
models and stayed there. The paused-board cache retained at most one scaled
surface and one RGB array; frame blending retained one float array. Observed
effects stayed at or below 24 and notices at or below three. At the larger
window, cached-board/blend storage reached
2,190,400 / 6,912 / 27,648 bytes respectively; this is not total process memory.

Weak-reference checks verified reclamation, per process, of 1,440 retired state
tensors, 606 scaled surfaces, 606 cached RGB arrays, 360 blend arrays, 120 duel
controllers and 120 lesson controllers. The five models, UI and arena were also
reclaimable at teardown. Object reclamation does not imply that every native
allocator reservation returns immediately to the operating system.

## Scope limits

The diagnostics cover **70,042 simulation ticks** across the six measured
processes. They do not cover all seed/checkpoint combinations, normal wall-clock
frame pacing, long training, native-display resources, other operating systems
or accelerators. Explicit GC and diagnostic scalar records affect memory
observations. Native desktop follow-up was still blocked by a disconnected
desktop; dummy-SDL results do not substitute for that manual QA.
