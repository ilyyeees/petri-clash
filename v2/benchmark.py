"""CPU-only, unpaced simulation benchmark with machine-readable output.

Examples (from the repository root)::

    python v2/benchmark.py --mode hard --steps 200 --output benchmark.json
    python v2/benchmark.py --mode nca --synthetic --cpu-threads 4

Trained models are loaded with the application's checkpoint selection policy.
Synthetic mode is explicit and never represented as trained-model performance.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys

import torch

from nca import NCA, make_seed
from runtime import benchmark_steps, configure_runtime


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("nca", "soft", "hard"), default="hard")
    parser.add_argument("--grid-size", type=nonnegative_int, default=0, help="0 uses checkpoint grid size")
    parser.add_argument("--left", type=positive_int, default=1, help="1-based target index")
    parser.add_argument("--right", type=positive_int, default=2, help="1-based target index")
    parser.add_argument("--left-seed", type=nonnegative_int)
    parser.add_argument("--right-seed", type=nonnegative_int)
    parser.add_argument("--seed", type=nonnegative_int, default=0, help="simulation RNG seed")
    parser.add_argument("--cpu-threads", type=positive_int, help="default: PETRI_CPU_THREADS or 1")
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument("--output", type=Path, help="also save the JSON report to this file")
    parser.add_argument("--synthetic", action="store_true", help="explicitly use untrained models, no checkpoint loading")
    parser.add_argument("--channels", type=positive_int, default=24, help="synthetic model channels")
    parser.add_argument("--hidden-size", type=positive_int, default=256, help="synthetic model hidden width")
    return parser


def _synthetic_bundle(args):
    if args.channels < 5:
        raise ValueError("synthetic models need at least 5 channels for the alpha and hidden seed")
    model = NCA(channels=args.channels, hidden_size=args.hidden_size).cpu().eval()
    return {
        "model": model,
        "channels": args.channels,
        "grid_size": args.grid_size or 48,
        "kind": "synthetic-untrained",
        "source": "synthetic-untrained",
    }


def _load_bundles(args):
    count = 1 if args.mode == "nca" else 2
    if args.synthetic:
        return [_synthetic_bundle(args) for _ in range(count)]
    # Keep checkpoint loading and optional UI dependencies out of the synthetic
    # microbenchmark. Loader messages go to stderr so stdout is valid JSON.
    with redirect_stdout(sys.stderr):
        from clash import ensure_model, list_targets

        targets = list_targets()
        selections = [(args.left, args.left_seed), (args.right, args.right_seed)][:count]
        bundles = []
        for index, preferred_seed in selections:
            if index > len(targets):
                raise ValueError(f"target index {index} is out of range; found {len(targets)} targets")
            bundle = ensure_model(targets[index - 1], "cpu", 0, preferred_seed=preferred_seed)
            if bundle.get("kind") == "scratch":
                raise ValueError("trained checkpoint unavailable; use --synthetic for an explicit untrained benchmark")
            bundles.append(bundle)
    return bundles


def _model_info(bundle):
    model = bundle["model"]
    result = {
        "kind": bundle.get("kind", "checkpoint"),
        "source": str(bundle.get("source", "unknown")),
        "channels": model.channels,
        "hidden_size": model.hidden_size,
        "fire_rate": model.fire_rate,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "training_grid_size": bundle.get("grid_size"),
    }
    source = Path(result["source"])
    if source.is_file():
        result["checkpoint_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


def run_benchmark(args):
    settings = configure_runtime(seed=args.seed, cpu_threads=args.cpu_threads)
    bundles = _load_bundles(args)
    size = args.grid_size or max(bundle["grid_size"] for bundle in bundles)
    if size < 5:
        raise ValueError("grid size must be at least 5")
    positions = [(size // 2, size // 2)] if args.mode == "nca" else [(size // 4, size // 2), (3 * size // 4, size // 2)]
    states = [
        make_seed(1, channels=bundle["channels"], height=size, xs=[pos[0]], ys=[pos[1]], device="cpu")
        for bundle, pos in zip(bundles, positions)
    ]
    owner = torch.zeros(1, 1, size, size, dtype=torch.long)
    control = torch.zeros(1, 1, size, size)
    if args.mode == "hard":
        from battle import clash_step

        for side, (x, y) in enumerate(positions, 1):
            owner[0, 0, y, x] = side
            control[0, 0, y, x] = 1 if side == 1 else -1

    def step():
        nonlocal states, owner, control
        if args.mode == "hard":
            a, b, owner, control = clash_step(*states, owner, control, bundles[0]["model"], bundles[1]["model"])
            states = [a, b]
        else:
            states = [bundle["model"](state, steps=1) for bundle, state in zip(bundles, states)]

    timing = benchmark_steps(step, warmup=args.warmup, steps=args.steps)
    final_state = []
    for state in states:
        final_state.append({
            "alive_cells": int((state[:, 3:4] > 0.1).sum().item()),
            "finite": bool(torch.isfinite(state).all().item()),
            "sha256": hashlib.sha256(state.contiguous().numpy().tobytes()).hexdigest(),
        })
    report = {
        "schema_version": 1,
        "benchmark": "petri-clash-cpu-simulation",
        "mode": args.mode,
        "synthetic": args.synthetic,
        "runtime": settings,
        "grid": {"height": size, "width": size, "batch_size": 1},
        "models": [_model_info(bundle) for bundle in bundles],
        "initial_positions": positions,
        "measurement": {
            "unit": "one simulation step",
            "includes": "neural update and battle rules" if args.mode == "hard" else "neural update",
            "excludes": ["model loading", "warmup", "rendering", "event handling", "FPS pacing", "final diagnostics"],
        },
        **timing,
        "final_state": final_state,
    }
    if args.mode == "hard":
        report["final_owned_cells"] = [int((owner == team).sum().item()) for team in (1, 2)]
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args)
        serialized = json.dumps(report, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"benchmark: {exc}\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
