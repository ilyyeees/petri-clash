import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from v2.common import model_from_config, pick_device, save_json, setup_torch, target_tensor_from_config
from v2.train_v2 import evaluate_model, resolve_config


def load_checkpoint(path, config, device):
    blob = torch.load(path, map_location=device)
    model = model_from_config(config, device)
    model.load_state_dict(blob["model"])
    return model, blob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v2/configs/rtx5090_base.toml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-dir")
    parser.add_argument("--target")
    parser.add_argument("--device")
    parser.add_argument("--out")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.device:
        overrides.setdefault("runtime", {})["device"] = args.device
    config = resolve_config(args.config, overrides)
    device = pick_device(config["runtime"].get("device", "auto"))
    setup_torch(config["runtime"], device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    run_dir = Path(args.run_dir) if args.run_dir else None
    target_path = Path(args.target) if args.target else None

    if run_dir is not None:
        checkpoint_path = checkpoint_path or run_dir / "checkpoints" / "best.pt"
        resolved = json.loads((run_dir / "resolved_config.json").read_text())
        target_path = target_path or Path(resolved["target"])

    if checkpoint_path is None or target_path is None:
        raise SystemExit("need either --run-dir or both --checkpoint and --target")

    model, blob = load_checkpoint(checkpoint_path, config, device)
    target = target_tensor_from_config(target_path, config, device)
    out_path = Path(args.out) if args.out else checkpoint_path.with_suffix(".eval.png")
    summary = evaluate_model(model, target, config, device, out_path=out_path)
    summary["checkpoint"] = str(checkpoint_path)
    summary["target"] = str(target_path)
    summary["checkpoint_step"] = int(blob.get("step", -1))
    save_json(out_path.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
