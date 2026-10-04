# Paired duel diagnostic

Run from the repository root:

```bash
python v2/compare.py --left 1 --right 6 --seeds 0,7,42 --device cpu --report comparison.json
```

With no options, this runs heart versus star on CPU: three simulation seeds
(`0,7,42`), two orientations per seed, 600 ticks per round including 60 unscored
warmup ticks. `--mode soft`, `--grid-size`, `--cpu-threads`, `--round-ticks` and
`--warmup-ticks` are available. `--left-seed` selects A's checkpoint seed and
`--right-seed` selects B's checkpoint seed; those choices follow the culture
when its side changes. There is no training/bootstrap or arbitrary checkpoint
path option. Only the existing ready-checkpoint policy is used.

A and B are distinct comparison labels even when both select the same culture.
Each seed runs A-left/B-right and then B-left/A-right, using a fresh Arena with
the same simulation seed, mode, grid, current rules, duration and mirrored
positions. The tool does not change the dynamics or imply that mirrored
placement makes the dynamics fair. Loading and simulation do not import pygame,
initialize SDL, render, pace frames, train, or use paid compute.

## Interpreting results

Each pair maps left/right cell-ticks back to A/B and sums both legs. The overall
score sums those paired scores, rather than counting per-seed wins. Outcomes are
selected using exact integers, including an exact draw; display averages from
the underlying duel do not select a winner. Hard mode scores held cells; soft
mode scores independent living cells, including overlap, rather than territory.

The terminal prints concise progress and "Paired score favors A/B" (or an exact
draw). A small seeded comparison is not a general ranking, significance test,
or balance claim. Winner changes after side swaps are reported explicitly,
including changes to/from a draw. They are evidence to inspect, not proof of a
particular cause.

JSON is written only with `--report`. Schema `petri-clash-paired-duel`, version 1,
includes portable culture names and selected checkpoint seed folders, software
and runtime settings, rules and starting positions, raw duel results, per-seed
summaries, and overall totals. `leg_winners` are A/B/draw in A-left then B-left
order; `leg_indices` point to those raw legs. There are no absolute checkpoint
paths or checkpoint fingerprints. Reproducing across another software version,
device or changed checkpoint bytes is not promised.

The first failed or non-finite leg stops the run. Its raw/partial diagnostic
result is retained, and that pair receives no combined score or outcome. An
incomplete or invalid comparison has null overall `scores` and `paired_outcome`.
`completed_pair_scores` are explicitly partial diagnostics and do not establish
an overall winner. Failed model exceptions retain their stage/type in JSON;
full exception text stays in the terminal because it can contain host paths.
Exit status is 0 for a complete comparison, 1 for a failed/invalid leg, and 2 for
invalid input or a report-writing error.

## Verification (2026-10-04)

Tests use tiny synthetic models with the real Arena and exact DuelRound scorer.
They cover both modes, repeated simulation seeds, distinct seed-dependent
results, preserved checkpoint selection across orientations, same-culture A/B
labels, huge integer ties and one-cell-tick margins, input/export independence,
invalid terminal evidence, model failures and non-finite results, late failures,
model-identity drift, CLI validation, and subprocess import guards for pygame
and SDL. A short real checkpoint pair is enabled by `PETRI_TEST_CHECKPOINTS=1`.

```bash
PETRI_TEST_CHECKPOINTS=1 python -m pytest v2/tests/test_compare.py -q
python -m compileall -q v2/compare.py v2/tests/test_compare.py
```

A six-round shipped-checkpoint CPU run used heart `seed_000` and flower
`seed_002`, a 48×48 grid, starting positions (16,23)/(31,23), standard rules,
600/60 ticks, one intra-op CPU thread, Python 3.12.14, Torch 2.14.1+cpu and
NumPy 2.3.5. Results:

| Simulation seed | A: heart cell-ticks | B: flower cell-ticks | Leg winners | Paired outcome |
| --- | ---: | ---: | --- | --- |
| 0 | 240,320 | 249,956 | B / B | B |
| 7 | 242,161 | 247,747 | A / B | B |
| 42 | 242,975 | 247,898 | B / B | B |
| Total | 725,456 | 745,601 | | B |

This reproduces the seed-7 winner change under a side swap. The paired score
favors flower for these seeds and settings; it does not establish general model
quality or fair/balanced dynamics. All six rounds completed with finite states.

A second full six-round run reproduced every raw duel result and the complete
summary exactly. Only timing fields differ. The final focused suite passed
24 tests plus 41 subtests with shipped-checkpoint opt-in, and compile checks
passed. No GPU run or training was performed.
