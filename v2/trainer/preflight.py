import argparse
from pathlib import Path

import torch

from trainer.common import load_config, merge_dict, pick_device, save_json
from trainer.train_v2 import train_target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="trainer/configs/single_gpu_base.toml")
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    base = load_config(args.config)
    target = Path(args.target) if args.target else sorted(Path(".").glob(base["run"]["target_glob"]))[0]
    device = pick_device(base["runtime"].get("device", "auto"))

    overrides = {
        "run": {"group_name": f"preflight_{base['run']['group_name']}"},
        "runtime": {"device": device},
        "train": {
            "steps": 8,
            "pool_size": min(64, base["train"]["pool_size"]),
            "batch_size": min(4, base["train"]["batch_size"]),
            "save_every": 4,
            "eval_every": 4,
            "preview_every": 4,
        },
        "eval": {
            "steps": [8, 16],
            "preview_steps": [8, 16],
            "damage_after": 8,
            "recover_steps": 8,
            "recovery_preview_steps": [0, 4, 8],
        },
    }
    if device != "cuda":
        overrides["train"]["compile"] = False
        overrides["train"]["amp"] = False
    config = merge_dict(base, overrides)
    run_dir = train_target(config, target, args.seed)

    latest = run_dir / "checkpoints" / "latest.pt"
    best = run_dir / "checkpoints" / "best.pt"
    previews = sorted((run_dir / "previews").glob("*.png"))
    result = {
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "run_dir": str(run_dir),
        "latest_exists": latest.exists(),
        "best_exists": best.exists(),
        "preview_count": len(previews),
    }
    save_json(run_dir / "preflight_result.json", result)
    print(result)


if __name__ == "__main__":
    main()
