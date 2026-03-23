# petri clash

two neural cellular automata grow on the same grid and fight over it.

## run it

```bash
conda env create -f environment.yml
conda activate petri-clash
python clash.py
```

on a fresh repo, `clash.py` will do a quick bootstrap train for any missing weights and save them in `weights/`.
that bootstrap now goes through the v2 trainer and exports a seed folder like `weights/01_heart/seed_000/`.

for a better organism, train a target properly:

```bash
python train.py --target targets/01_heart.png --steps 3000
```

`train.py` is now just a single-target wrapper around the `trainer/` stack, so it writes the same v2-format weights that `clash.py` expects.

## big gpu training

for remote long-horizon training on a single rented gpu, use the `trainer/` stack inside this folder.

the setup and workflow are in `trainer/README.md`.

that stack now uses resumable checkpoints, a cosine lr schedule, tensorboard logs, and shell wrappers built around `python -m trainer...` entrypoints.

## current training status

the full all-target gpu run artifacts are saved under `weights/` and the run-level metadata is under `trainer_artifacts/single_gpu_all_targets/`.

targets with a usable accepted seed:

- `01_heart` -> use `weights/01_heart/seed_000/`
- `02_star` -> use `weights/02_star/seed_001/`
- `03_sun` -> use `weights/03_sun/seed_002/`
- `06_flower` -> use `weights/06_flower/seed_002/`
- `07_umbrella` -> use `weights/07_umbrella/seed_000/`

what actually converged:

- `02_star/seed_001` is the only run that hit the explicit `converged` stop and ended early

accepted but ran the full budget instead of hitting the explicit convergence stop:

- `01_heart/seed_000`
- `03_sun/seed_002`
- `06_flower/seed_002`
- `07_umbrella/seed_000`

failed targets:

- `04_moon` -> `seed_000`, `seed_001`, and `seed_002` all collapsed
- `05_bolt` -> `seed_000`, `seed_001`, and `seed_002` all collapsed
- `08_yin` -> `seed_000`, `seed_001`, and `seed_002` all collapsed
- `09_skull` -> `seed_000`, `seed_001`, and `seed_002` all collapsed

one important caveat:

- `03_sun/seed_002` was accepted by the pipeline, but it is much weaker than heart, star, flower, and umbrella and does not fully nail the target shape

if you want the exact final run summary, open `trainer_artifacts/single_gpu_all_targets/group_summary.json` and `trainer_artifacts/single_gpu_all_targets/logs/train_all-20260322-210038.log`.

## controls

- `space` pause / resume
- `r` reset seed positions
- `1` to `9` pick the left organism target
- `shift+1` to `shift+9` pick the right organism target
- left click drops a damage crater
- `esc` quits
