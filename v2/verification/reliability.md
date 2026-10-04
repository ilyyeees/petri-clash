# Checkpoint and export reliability

Verified on 2026-10-04 as a consolidation of the cumulative v2 changes. The
simulation rules, model parameters and `v1/` remain unchanged. This pass addresses
specific malformed-file recovery and final-export mismatches found during
integration review rather than adding another game mode.

## Reject malformed play data before replacing a culture

Previously, some truncated local checkpoints raised loader exceptions that the
interface did not handle. Malformed optional health metadata could also fail
while drawing an unrelated culture's picker status. A four-channel checkpoint
could load successfully, then fail while constructing the arena's required
hidden seed channel after the active culture had already been changed.

The changes keep validation at the input boundaries:

- Expected restricted-deserialization failures become a recoverable `ValueError`,
  with their original cause retained for diagnostics.
- Both the normal and historical NumPy-RNG loading attempts still explicitly use
  `weights_only=True`; no unrestricted fallback or new allowlisted objects are added.
- Optional health configuration and its `stop` section are checked as mappings.
  Invalid optional threshold metadata keeps the existing default of 0.2; missing
  evaluation evidence still means unverified rather than ready.
- Play configuration checks model/data/train sections and scalar settings before
  model construction. It requires at least five channels, a positive hidden size,
  a grid of at least eight, and a finite fire rate from zero through one.
- Unsupported signed-64-bit grid dimensions/storage sizes are rejected before
  loading the playable bundle. This is a representation bound, not a RAM budget.
- Numeric and model-state validation notices do not echo unbounded saved values;
  the original exception remains available as the cause.
- Normalized parameter names and tensor shapes must match that configuration before
  allocating the NCA. Plain, compiled-wrapper and legacy checkpoints remain supported.
- Malformed state-dict module metadata is rejected with a clear input error.

Selections rejected by these checks retain the running match and show a notice. Deferred opponents
that cannot be loaded still use the existing visible lesson-exit fallback while
keeping the requested checkpoint pin. Loader validation does not retrain,
substitute or rewrite weights.

Only load trusted checkpoint files. These are application-recovery checks, not a
claim that arbitrary files are safe or that sufficient memory is available.
PyTorch documents limitations of the
[restricted weights-only loader](https://docs.pytorch.org/docs/2.14/notes/serialization.html#torch-load-with-weights-only-true).
Ordinary resource and device failures are not converted into successful loads.

## Capture the final state on exit

A queued STEP followed by Quit used to end the simulation at tick 1 but save the
last drawn tick-0 screen. An explicit interactive `--snapshot` now clears only
the presentation blend and draws the current state with zero animation elapsed
before saving. This makes the board and HUD agree with the final report, even
when the last action and exit arrive in the same event batch.

The final draw never advances the simulation or its RNG. Without a requested
snapshot there is no additional exit draw. Existing reduced-motion settings,
queued resize/color changes and bounded `--ui-frames` runs retain their behavior.

## Safer instructions and shareable reports

The optional training example now exports to `user-weights` and evaluates that
same experiment, instead of silently replacing the shipped heart seed-0 export.
The documentation states that experiments are not automatically activated by the
arena and that omitting the export-root option replaces the matching bundled
export. Trainer checkpoint writes are atomic; the separate export-copy operation
is not transactional. The documented training example was not executed, and no bundled weights were
retrained or exported.

Both quick starts identify the Apple-silicon wheel choice before the installation
commands, following the [official PyTorch installation guidance](https://pytorch.org/get-started/locally/).
Fresh macOS/Windows installation was not tested here.

Ordinary diagnostic match and benchmark reports retain local paths, and benchmarks
retain fingerprints for private reproducibility work. The README now warns to
review/sanitize those before sharing and points to the portable comparison and
recipe/replay formats. The three original public CPU timing JSON files omit
checkpoint/state fingerprints; all their other measurements and configurations
were checked unchanged. This does not remove anything from historical Git commits.

## Integrated verification

The complete CPU suite passed **628 tests and 1,121 subtests**, with shipped
checkpoint checks enabled and no skips. The new recovery and final-snapshot
suites contribute 224 cases. Compilation, whitespace checks and all 56 local
documentation links passed.

The checks include temporary malformed-file fixtures, the real restricted loader,
and dummy-SDL lifecycle tests. They cover rejection via mouse/keyboard in lab,
duel and lesson, unchanged active bundles/indices/tensors/ticks/pins/cache and all
three CPU RNGs, and usable deferred-opponent fallback.

All 23 shipped checkpoint bundles were also compared with the prior loader:
parameters and loaded settings remained exact, followed by 16 exact state/RNG
steps per checkpoint. This includes the collapsed research checkpoints; loading
compatibility is not a claim that their learned growth is healthy.

The existing inference audit was rerun across heart/star and heart/flower in hard
and soft modes. Four full 600-tick duels retained exact state bytes/layouts, input
immutability, parameters, every RNG and every scoring snapshot at each tick.

A bounded mixed-action check replayed 16 seeded schedules twice, totaling 3,840
actions across pause/step/reset, views, culture selections, mouse tools,
modifiers, resizes, lessons and duels. Every checkpoint had coherent tensor
shapes and current HUD counts; each repeated schedule ended with identical
simulation/RNG state and SDL pixels. This is regression sampling, not exhaustive
input coverage or a native-window test.

Post-change headless checks with a deliberately invalid SDL driver passed:

- Maximum simulation seed 4,294,967,295, with 65 requested ticks stopping at the
  exact 60-tick endpoint; 52 scoring ticks, 6,725 / 4,377 cell-ticks
- Portable export and rerun of that round with the same exact scores
- Four side-swapped comparison legs over seeds 7 and 42 at 64/8 ticks
- The existing 600-tick heart/flower example, still 123,178 / 121,494 cell-ticks

Short comparison outcomes are not general rankings. All execution here was Linux
CPU; native-window, assistive-technology and accelerator behavior remain untested.

## Reproduce

```bash
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests -q
python -m compileall -q v2
python v2/replay.py v2/verification/duel-recipe-example.json
python v2/verification/benchmark_inference.py --steps 160 --frame-steps 320 \
  --warmup 30 --repeats 2 --replay-ticks 600 --replay-warmup 60 \
  --output inference-recheck.json
```

The inference script needs its documented baseline Git revision locally. It
measures simulation rather than native display or universal FPS; hardware and
competing host work affect timings.
