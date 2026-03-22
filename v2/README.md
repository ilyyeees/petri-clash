# v2 training stack

this is the remote-first training stack for long-horizon nca work.

the main idea is simple:

- keep the playable baseline in the root scripts
- do heavy training and evaluation through `v2/`
- make every run resumable
- make every run exportable as a small bundle
- optimize for a box like an rtx 5090 instead of a laptop cpu

## what is here

- `train_v2.py` trains one target with stronger losses and longer rollout curriculum
- `train_all_v2.py` trains every target and seed from one config
- `eval_v2.py` scores a checkpoint and saves a preview image
- `preflight.py` runs a tiny smoke test on the current machine
- `bundle_run.py` packs the useful outputs into a small tarball
- `configs/rtx5090_base.toml` is the default all-target config
- `scripts/` contains the shell wrappers for remote use

## why this exists

the original stack was good enough to prove the idea.

it was not good enough for long-horizon stability, remote runs, or all-target sweeps on a big gpu.

this stack changes that by doing a few specific things:

- it adds stronger penalties for off-target alpha drift
- it uses a rollout curriculum so the model gets pushed to survive longer
- it evaluates checkpoints at longer horizons instead of trusting only training loss
- it saves a best checkpoint and a resumable latest checkpoint
- it writes previews and metrics automatically

## expected remote workflow

on a fresh vast machine:

```bash
git clone git@github.com:ilyyeees/petri-clash.git
cd petri-clash
git checkout v2-rtx5090-training
bash v2/scripts/bootstrap_vast.sh
bash v2/scripts/preflight.sh
bash v2/scripts/launch_remote.sh
```

to monitor:

```bash
bash v2/scripts/status.sh
```

to make a compact archive of the best outputs:

```bash
bash v2/scripts/bundle_group.sh
```

if you launch a run with a custom group name, pass the matching group dir when you bundle:

```bash
bash v2/scripts/bundle_group.sh v2/configs/rtx5090_base.toml --group-dir runs_v2/my_custom_group
```

## results layout

the outputs land under:

```text
runs_v2/<group>/<target>/seed_<n>/
```

inside each run you will get:

- `resolved_config.json`
- `metrics.jsonl`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- preview pngs
- `best_summary.json`

the archive script only grabs the important pieces so retrieval stays fast.

## notes for a 5090 box

- use a pytorch image that already has a modern cuda build
- this stack expects cuda for amp and compile to matter
- the config is tuned for a big single gpu, not for cpu fallback
- if you want bigger arenas, retrain at that arena size instead of reusing 48x48 weights
