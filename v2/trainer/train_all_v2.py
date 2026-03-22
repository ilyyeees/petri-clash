import argparse
import json
from pathlib import Path

from trainer.common import list_target_paths, project_path, run_dir_for, save_json
from trainer.train_v2 import resolve_config, train_target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="trainer/configs/single_gpu_base.toml")
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--group-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_best_score(run_dir):
    path = Path(run_dir) / "best_summary.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    return float(blob["score"]), blob


def load_run_stop(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return None

    latest = None
    for line in path.open():
        blob = json.loads(line)
        if blob.get("kind") == "run_stop":
            latest = blob
    return latest


def seed_order(config, args):
    if args.seeds:
        return [int(seed) for seed in args.seeds]

    primary = [int(seed) for seed in config["run"].get("seeds", [0])]
    fallback = [int(seed) for seed in config["run"].get("fallback_seeds", [])]
    ordered = []
    for seed in primary + fallback:
        if seed not in ordered:
            ordered.append(seed)
    return ordered


def run_status(run_dir):
    best = load_best_score(run_dir)
    stop = load_run_stop(run_dir)
    return {
        "best": best,
        "stop": stop,
        "collapsed": stop is not None and stop.get("reason") == "collapsed",
    }


def write_group_summary(config, target_paths, seeds):
    group_root = project_path(config["run"]["output_root"]) / config["run"]["group_name"]
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
    if args.group_name:
        overrides.setdefault("run", {})["group_name"] = args.group_name
    if args.batch_size is not None:
        overrides.setdefault("train", {})["batch_size"] = args.batch_size
    if args.pool_size is not None:
        overrides.setdefault("train", {})["pool_size"] = args.pool_size
    if args.no_compile:
        overrides.setdefault("train", {})["compile"] = False
    if args.no_amp:
        overrides.setdefault("train", {})["amp"] = False

    config = resolve_config(args.config, overrides)
    target_paths = list_target_paths(config)
    if args.targets:
        chosen = set(args.targets)
        target_paths = [path for path in target_paths if path.stem in chosen or path.name in chosen]
    seeds = seed_order(config, args)

    print(f"targets: {[path.stem for path in target_paths]}")
    print(f"seed order: {seeds}")

    for target_path in target_paths:
        accepted_seed = None
        for seed in seeds:
            print(f"training {target_path.stem} seed {seed}")
            run_dir = train_target(config, target_path, seed)
            write_group_summary(config, target_paths, seeds)
            status = run_status(run_dir)

            if status["collapsed"]:
                print(f"{target_path.stem} seed {seed} collapsed, trying the next seed")
                continue

            if status["best"] is not None:
                print(f"{target_path.stem} accepted seed {seed} with score {status['best'][0]:.5f}")
                accepted_seed = seed
                break

            print(f"{target_path.stem} seed {seed} produced no best checkpoint, trying the next seed")

        if accepted_seed is None:
            print(f"{target_path.stem} finished without a valid seed")

    print("done")


if __name__ == "__main__":
    main()
