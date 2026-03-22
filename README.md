# petri clash

two neural cellular automata grow on the same grid and fight over it.

## run it

```bash
conda env create -f environment.yml
conda activate petri-clash
python clash.py
```

on a fresh repo, `clash.py` will do a quick bootstrap train for any missing weights and save them in `weights/`.

for a better organism, train a target properly:

```bash
python train.py --target targets/01_heart.png --steps 3000
```

## controls

- `space` pause / resume
- `r` reset seed positions
- `1` to `9` pick the left organism target
- `shift+1` to `shift+9` pick the right organism target
- left click drops a damage crater
- `esc` quits
