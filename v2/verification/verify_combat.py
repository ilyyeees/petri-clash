"""Reproduce the CPU measurements described in verification/README.md."""

import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'v2'))
import torch
from nca import NCA, make_seed
from checkpoints import load_checkpoint_file, load_model_state
from battle import clash_step, crater
root = ROOT / 'v2'
torch.set_num_threads(1)

def load(target, seed):
    blob = load_checkpoint_file(root / 'weights' / target / f'seed_{seed:03d}' / 'checkpoints' / 'best.pt', map_location='cpu')
    cfg = blob['config']['model']
    model = NCA(**{k: cfg[k] for k in ('channels', 'hidden_size', 'fire_rate')})
    load_model_state(model, blob['model'])
    return (model.eval(), cfg['channels'])
a_model, a_ch = load('01_heart', 0)
b_model, b_ch = load('02_star', 1)
torch.manual_seed(42)
a = make_seed(1, channels=a_ch, height=48, xs=[20], ys=[24])
b = make_seed(1, channels=b_ch, height=48, xs=[28], ys=[24])
owner = torch.zeros(1, 1, 48, 48, dtype=torch.long)
owner[0, 0, 24, 20] = 1
owner[0, 0, 24, 28] = 2
control = torch.zeros(1, 1, 48, 48)
control[0, 0, 24, 20] = 1
control[0, 0, 24, 28] = -1
captures = 0
releases = 0
snapshots = []
start = time.perf_counter()
for step in range(1, 513):
    previous = owner
    a, b, owner, control = clash_step(a, b, owner, control, a_model, b_model)
    captures += int(((previous == 0) & (owner > 0)).sum())
    releases += int(((previous > 0) & (owner == 0)).sum())
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(control).all()
    assert control.abs().max() <= 1
    assert (a * (owner == 2)).abs().sum() == 0 and (b * (owner == 1)).abs().sum() == 0
    if step in (64, 128, 256, 257, 320, 384, 512):
        snapshots.append({'step': step, 'alive_a': int((a[:, 3:4] > 0.1).sum()), 'alive_b': int((b[:, 3:4] > 0.1).sum()), 'owned_a': int((owner == 1).sum()), 'owned_b': int((owner == 2).sum()), 'control_max': float(control.abs().max())})
    if step == 256:
        a, b, owner, control = crater(a, b, owner, control, 24, 24, 4)
        snapshots.append({'step': '256_after_crater', 'alive_a': int((a[:, 3:4] > 0.1).sum()), 'alive_b': int((b[:, 3:4] > 0.1).sum()), 'owned_a': int((owner == 1).sum()), 'owned_b': int((owner == 2).sum()), 'control_max': float(control.abs().max())})
result = {'match': '01_heart seed000 vs 02_star seed001', 'positions': [[20, 24], [28, 24]], 'random_seed': 42, 'device': 'cpu', 'threads': 1, 'grid': 48, 'steps': 512, 'seconds': round(time.perf_counter() - start, 4), 'captures': captures, 'releases': releases, 'snapshots': snapshots, 'checks': 'All 512 ticks finite and no surviving enemy state in opposing owned territory; control in [-1,1]. Crater at (24,24), radius4, after tick256.'}
print(json.dumps(result, indent=2))
Path(__file__).with_name('combat.json').write_text(json.dumps(result, indent=2) + '\n')
