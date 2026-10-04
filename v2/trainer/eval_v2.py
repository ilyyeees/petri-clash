import argparse
import json
from pathlib import Path

from checkpoints import checkpoint_config, load_checkpoint_file, load_model_state
from trainer.common import (
    merge_dict,
    model_from_config,
    pick_device,
    project_path,
    save_json,
    setup_torch,
    target_tensor_from_config,
)
from trainer.train_v2 import evaluate_model, resolve_config


def load_checkpoint(path, config, device):
    """Load using saved model settings rather than the caller's defaults."""
    blob = load_checkpoint_file(path, map_location="cpu")
    config = checkpoint_config(blob, config, checkpoint_path=path)
    model = model_from_config(config, device)
    load_model_state(model, blob["model"])
    model.eval()
    return model, blob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="trainer/configs/single_gpu_base.toml",
                        help="fallback settings for fields missing from the saved config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-dir")
    parser.add_argument("--target", help="override the checkpoint's saved target path")
    parser.add_argument("--device")
    parser.add_argument("--out")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is not None:
        checkpoint_path = checkpoint_path or run_dir / "checkpoints" / "best.pt"
    if checkpoint_path is None:
        raise SystemExit("need --run-dir or --checkpoint")

    # Resolve the complete saved experiment before creating the model or target.
    # Defaults only fill old/missing fields; --device remains an explicit override.
    defaults = resolve_config(args.config)
    if run_dir is not None:
        resolved_path = run_dir / "resolved_config.json"
        if resolved_path.is_file():
            defaults = merge_dict(defaults, json.loads(resolved_path.read_text()))
    blob = load_checkpoint_file(checkpoint_path, map_location="cpu")
    config = checkpoint_config(blob, defaults, checkpoint_path=checkpoint_path)
    if args.device:
        config.setdefault("runtime", {})["device"] = args.device
    device = pick_device(config["runtime"].get("device", "auto"))
    setup_torch(config["runtime"], device)

    target_value = args.target or config.get("target")
    if not target_value:
        raise SystemExit("checkpoint has no saved target; provide --target")
    target_path = Path(target_value)
    if not target_path.is_absolute() and not target_path.exists():
        target_path = project_path(target_path)

    model = model_from_config(config, device)
    load_model_state(model, blob["model"])
    model.eval()
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
