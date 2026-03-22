import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2.common import list_target_paths, run_dir_for, save_json
from v2.train_v2 import resolve_config, train_target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v2/configs/rtx5090_base.toml")
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_best_score(run_dir):
    path = Path(run_dir) / "best_summary.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    return float(blob["score"]), blob


def write_group_summary(config, target_paths, seeds):
    group_root = Path(config["run"]["output_root"]) / config["run"]["group_name"]
    summary = {"group": config["run"]["group_name"], "targets": {}}

    for target_path in target_paths:
        rows = []
        for seed in seeds:
            run_dir = run_dir_for(config, target_path, seed)
            score = load_best_score(run_dir)
            if score is None:
                continue
            rows.append(
                {
                    "seed": int(seed),
                    "score": score[0],
                    "run_dir": str(run_dir),
                    "summary": score[1],
                }
            )

        rows.sort(key=lambda row: row["score"])
        summary["targets"][target_path.stem] = {
            "runs": rows,
            "best": rows[0] if rows else None,
        }

    save_json(group_root / "group_summary.json", summary)


def main():
    args = parse_args()
    overrides = {}
    if args.device:
        overrides.setdefault("runtime", {})["device"] = args.device
    if args.steps is not None:
        overrides.setdefault("train", {})["steps"] = args.steps
    if args.no_compile:
        overrides.setdefault("train", {})["compile"] = False
    if args.no_amp:
        overrides.setdefault("train", {})["amp"] = False

    config = resolve_config(args.config, overrides)
    target_paths = list_target_paths(config)
    if args.targets:
        chosen = set(args.targets)
        target_paths = [path for path in target_paths if path.stem in chosen or path.name in chosen]
    seeds = args.seeds or [int(seed) for seed in config["run"]["seeds"]]

    print(f"targets: {[path.stem for path in target_paths]}")
    print(f"seeds: {seeds}")

    for target_path in target_paths:
        for seed in seeds:
            print(f"training {target_path.stem} seed {seed}")
            train_target(config, target_path, seed)
            write_group_summary(config, target_paths, seeds)

    print("done")


if __name__ == "__main__":
    main()
