import argparse
import tarfile
from pathlib import Path

from v2.common import load_config, now_stamp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v2/configs/single_gpu_base.toml")
    parser.add_argument("--run-dir")
    parser.add_argument("--group-dir")
    parser.add_argument("--out")
    return parser.parse_args()


def latest_preview(run_dir):
    previews = sorted((Path(run_dir) / "previews").glob("*.png"))
    return previews[-1] if previews else None


def files_for_run(run_dir):
    run_dir = Path(run_dir)
    paths = [
        run_dir / "resolved_config.json",
        run_dir / "best_summary.json",
        run_dir / "metrics.jsonl",
        run_dir / "checkpoints" / "best.pt",
    ]
    preview = latest_preview(run_dir)
    if preview is not None:
        paths.append(preview)
    return [path for path in paths if path.exists()]


def files_for_group(group_dir):
    group_dir = Path(group_dir)
    paths = []
    summary = group_dir / "group_summary.json"
    if summary.exists():
        paths.append(summary)
    for run_dir in sorted(group_dir.glob("*/*")):
        if run_dir.is_dir():
            paths.extend(files_for_run(run_dir))
    return paths


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.run_dir:
        source = Path(args.run_dir)
        files = files_for_run(source)
        default_name = f"{source.name}-{now_stamp()}.tar.gz"
        base_dir = source
    else:
        source = Path(args.group_dir) if args.group_dir else Path(config["run"]["output_root"]) / config["run"]["group_name"]
        files = files_for_group(source)
        default_name = f"{source.name}-{now_stamp()}.tar.gz"
        base_dir = source.parent

    if not files:
        raise SystemExit(f"nothing to bundle under {source}")

    out_path = Path(args.out) if args.out else Path("bundles_v2") / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_path, "w:gz") as handle:
        for path in files:
            handle.add(path, arcname=str(path.relative_to(base_dir)))

    print(out_path)


if __name__ == "__main__":
    main()
