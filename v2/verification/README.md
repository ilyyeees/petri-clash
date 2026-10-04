# Verification

Verification for the cumulative v2 improvements:

- [Mouse-only lab tools](lab-tools.md): visible planting, damage, radius and clear controls with shared pointer previews
- [Checkpoint picker consistency](checkpoint-picker.md): side-specific seed pins, honest loaded identity, and lesson fallback intent
- [Portable duel recipes](duel-recipes.md): saving completed rounds, bounded CPU reruns, and explicit score checks
- [CPU inference efficiency](inference-efficiency.md): bounded activation reuse, exact replay checks, and paired latency measurements
- [Guided regrowth lesson](lesson.md): interactive flow, recovery measurements, and visual checks
- [Paired seeded comparisons](compare.md): both side assignments, exact score mapping, and repeatability
- [Raster efficiency](raster-efficiency.md): reduced drawing work with identical pixels
- [Seeded duels](duels.md): round lifecycle, scoring, and replay
- [Tactical interface and live feedback](feedback.md): score readability, animation, and controls

## Original living arena upgrade

Verified locally on 2026-10-04 against baseline commit `9a941a7c2b283d3acb586009dd14801174438761`. Python 3.12, CPU-only PyTorch 2.14.1. Existing model weights and `v1/` are unchanged.

### Regression coverage

The full run passed **67 tests and 37 subtests**, with `PETRI_TEST_CHECKPOINTS=1`; no skipped test in that run. Syntax compilation and `git diff --check` also passed.

Coverage includes delayed frontier maturation, team symmetry, stronger-neighbor takeover, finite-state quarantine, crater boundaries, dead land release, all five accepted pretrained organisms, all 23 shipped checkpoint files, plain/compiled compatibility, exact RNG/pool resume, atomic interrupted-save preservation, saved evaluation architecture, model health filtering, seeded reset, headless graphics sentinels, and repeated UI interactions.

Independent review also exercised real bundled models through pause, single-step, repeated reset/selection, failed collapsed selection, clear, planting after inference, damage, mode switches, resize, and a real saved-checkpoint evaluation. Review identified a test-runner mismatch and invalid CLI edge cases, which were corrected and retested.

A real tiny training run resumed from step 1 and saved step 2. `torch.compile` interoperability was checked using its eager CPU backend; full accelerator compilation and GPU training were not tested.

### Learned growth recovered

Each test below runs **one centered pretrained organism against an empty, dormant opponent**, on 48×48 cells for 256 ticks, one CPU thread, RNG seed 42. This isolates interference from the territory rules; it is not a win-rate or competitive tournament measurement. Each row uses exactly the same shipped checkpoint across all three rules.

Living cells use alpha > 0.1:

| Culture / checkpoint seed | Previous hard rules | Repaired hard rules | Solo NCA reference |
|---|---:|---:|---:|
| Heart / 000 | 180 | 242 | 243 |
| Star / 001 | 92 | 182 | 183 |
| Sun / 002 | 0 | 126 | 133 |
| Flower / 002 | 0 | 246 | 246 |
| Umbrella / 000 | 0 | 166 | 166 |

The corrected rules preserve the hidden state needed for a frontier to mature. Small differences from solo growth remain because hard mode still prunes unsupported tissue and applies ownership rules. [Raw growth measurements](growth.json).

### Actual opposing battle and damage

Heart seed 000 versus star seed 001, starting at (20,24)/(28,24), 48×48, seed 42, 512 CPU ticks:

- Every tick remained finite and within control range [-1,1]
- No surviving enemy hidden state inside opposing owned territory
- Center crater at tick 256, radius 4: living cells 195/139 → 159/130
- At tick 320: recovered to 188/141 living cells

This validates ongoing interaction and recovery, not strategic learning or balanced win rates. [Raw combat measurements](combat.json).

### Measured CPU thread policy

Matched runs of the **new** hard engine: same 24-channel/256-hidden heart/star models, same 48×48 grid and initial positions, RNG seed 0, 20 warmup ticks excluded, 100 timed ticks. Loading, rendering, event handling, frame pacing, and final diagnostics are excluded.

| CPU threads | Median tick | 95th percentile | Report |
|---|---:|---:|---|
| 1 | 7.43 ms | 8.79 ms | [JSON](cpu-1-thread.json) |
| 9 | 10.46 ms | 25.86 ms | [JSON](cpu-9-threads.json) |
| 1, repeat | 7.04 ms | 8.85 ms | [JSON](cpu-1-thread-repeat.json) |

The first one-thread run averaged 131.6 steps/s. Repeating with one thread produced identical final-state hashes. This supports a conservative one-thread default on this host; it is not a universal hardware result or an old-versus-new engine speedup. Measure your own CPU with `--cpu-threads`. No speculative NCA perception rewrite was retained.

### Reproduce

From the repository root with the development requirements installed:

```bash
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests -q
python -m compileall -q v2
python v2/verification/verify_growth.py
python v2/verification/verify_combat.py
python v2/benchmark.py --cpu-threads 1 --warmup 20 --steps 100 --output cpu-1.json
python v2/benchmark.py --cpu-threads 9 --warmup 20 --steps 100 --output cpu-9.json
python v2/clash.py --device cpu --headless-frames 256 --report match.json
SDL_VIDEODRIVER=dummy python v2/clash.py --device cpu --ui-frames 64 \
  --steps-per-frame 4 --fps 120 --snapshot arena.png
```

`verify_growth.py` reads the original rules from the baseline Git commit, so run it in a Git checkout with that commit available. Both verification scripts update their corresponding JSON files. Timings depend on host load. Screenshot rendering was inspected at default and minimum supported sizes.

### Limits

No GPU run, long training sweep, strategic self-play training, or v1 change is included. This upgrade fixes the arena/runtime around existing organisms; it does not make collapsed models trained. GitHub CI status is reported on the pull request separately from these local measurements.
