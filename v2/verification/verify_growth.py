"""Reproduce the CPU measurements described in verification/README.md."""

import ast, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'v2'))
import torch
import torch.nn.functional as F
from nca import NCA, make_seed
from checkpoints import load_checkpoint_file, load_model_state
from battle import clash_step, crater
root = ROOT
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
source = subprocess.check_output(['git', 'show', '9a941a7c2b283d3acb586009dd14801174438761:v2/clash.py'], cwd=root, text=True)
tree = ast.parse(source)
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('clash_step', 'momentum_owner') or (isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id.startswith('DEFAULT_') for t in n.targets)))]
ns = {'torch': torch, 'F': F, 'V2_ROOT': root / 'v2'}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<baseline_clash>', 'exec'), ns)
legacy_step = ns['clash_step']

class EmptyModel:

    def __call__(self, x, steps=1):
        return torch.zeros_like(x)

def load(target, seed):
    path = root / 'v2' / 'weights' / target / f'seed_{seed:03}' / 'checkpoints' / 'best.pt'
    blob = load_checkpoint_file(path, map_location='cpu')
    config = blob['config']
    cfg = config['model']
    model = NCA(channels=cfg['channels'], hidden_size=cfg['hidden_size'], fire_rate=cfg['fire_rate'])
    load_model_state(model, blob['model'])
    return (model.eval(), cfg['channels'], config['data']['grid_size'])
rows = []
for target, seed in [('01_heart', 0), ('02_star', 1), ('03_sun', 2), ('06_flower', 2), ('07_umbrella', 0)]:
    model, ch, size = load(target, seed)
    for mode in ['baseline_hard', 'frontier_hard', 'solo_nca']:
        torch.manual_seed(42)
        a = make_seed(1, channels=ch, height=size)
        b = torch.zeros_like(a)
        owner = torch.zeros(1, 1, size, size, dtype=torch.long)
        owner[0, 0, size // 2, size // 2] = 1
        control = owner.float()
        begin = time.perf_counter()
        snapshots = []
        with torch.inference_mode():
            for step in range(1, 257):
                if mode == 'solo_nca':
                    a = model(a, steps=1)
                else:
                    a, b, owner, control = (legacy_step if mode == 'baseline_hard' else clash_step)(a, b, owner, control, model, EmptyModel())
                if step in [32, 64, 128, 256]:
                    snapshots.append({'step': step, 'alive': int((a[:, 3:4] > 0.1).sum()), 'owned': int((owner == 1).sum()) if mode != 'solo_nca' else None, 'alpha_mass': float(a[:, 3:4].clamp(0, 1).sum()), 'finite': bool(torch.isfinite(a).all())})
        row = {'target': target, 'checkpoint_seed': seed, 'mode': mode, 'random_seed': 42, 'seconds': round(time.perf_counter() - begin, 4), 'snapshots': snapshots}
        rows.append(row)
        print(json.dumps(row), flush=True)
result = {'device': 'cpu', 'torch': torch.__version__, 'threads': 1, 'grid': 48, 'steps': 256, 'note': 'One pretrained organism against a dormant opponent, centered seed; seed 42 is reset for each mode. Existing checkpoint weights are never modified. Alive threshold is alpha > 0.1. Baseline is original commit 9a941a7 clash_step.', 'rows': rows}
Path(__file__).with_name('growth.json').write_text(json.dumps(result, indent=2) + '\n')
