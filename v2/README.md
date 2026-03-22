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

## controls

- `space` pause / resume
- `r` reset seed positions
- `1` to `9` pick the left organism target
- `shift+1` to `shift+9` pick the right organism target
- left click drops a damage crater
- `esc` quits
