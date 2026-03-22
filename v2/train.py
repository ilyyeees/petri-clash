import argparse
import json
import shutil
from pathlib import Path

from trainer.common import load_target
from trainer.train_v2 import resolve_config, train_target as run_single_target


DEFAULT_CONFIG = "trainer/configs/single_gpu_base.toml"


def best_preview_path(run_dir):
    summary_path = Path(run_dir) / "best_summary.json"
    if not summary_path.exists():
        return None

    try:
        step = int(json.loads(summary_path.read_text())["step"])
    except Exception:
        return None

    preview_path = Path(run_dir) / "previews" / f"step_{step:05d}_eval.png"
    if preview_path.exists():
        return preview_path
    return None


def export_run(run_dir, target_path, seed=0, export_root="weights"):
    run_dir = Path(run_dir)
    export_dir = Path(export_root) / Path(target_path).stem / f"seed_{int(seed):03d}"
    checkpoint_dir = export_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    if not best_checkpoint.exists():
        raise SystemExit(f"missing best checkpoint in {run_dir}")

    shutil.copy2(best_checkpoint, checkpoint_dir / "best.pt")

    for name in ["best_summary.json", "resolved_config.json"]:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, export_dir / name)

    preview_path = best_preview_path(run_dir)
    if preview_path is not None:
        shutil.copy2(preview_path, export_dir / preview_path.name)

    return checkpoint_dir / "best.pt"


def normalize_config(config):
    steps = int(config["train"]["steps"])
    config["train"]["save_every"] = max(1, min(int(config["train"]["save_every"]), steps))
    config["train"]["eval_every"] = max(1, min(int(config["train"]["eval_every"]), steps))

    preview_every = int(config["train"].get("preview_every", 0) or 0)
    if preview_every > 0:
        config["train"]["preview_every"] = max(1, min(preview_every, steps))

    return config


def train_target(
    target_path,
    steps=None,
    seed=0,
    device=None,
    config_path=DEFAULT_CONFIG,
    group_name=None,
    batch_size=None,
    pool_size=None,
    export_root="weights",
    no_compile=False,
    no_amp=False,
):
    config = resolve_config(config_path)

    if steps is not None:
        config["train"]["steps"] = int(steps)
    if device:
        config["runtime"]["device"] = device
    if group_name:
        config["run"]["group_name"] = group_name
    else:
        config["run"]["group_name"] = "single_target_train"
    if batch_size is not None:
        config["train"]["batch_size"] = int(batch_size)
    if pool_size is not None:
        config["train"]["pool_size"] = int(pool_size)
    if no_compile:
        config["train"]["compile"] = False
    if no_amp:
        config["train"]["amp"] = False

    config = normalize_config(config)
    run_dir = run_single_target(config, Path(target_path), int(seed))
    return export_run(run_dir, target_path, seed=seed, export_root=export_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--group-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--export-root", default="weights")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    checkpoint_path = train_target(
        args.target,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        config_path=args.config,
        group_name=args.group_name,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        export_root=args.export_root,
        no_compile=args.no_compile,
        no_amp=args.no_amp,
    )
    print(f"exported {checkpoint_path}")


if __name__ == "__main__":
    main()
