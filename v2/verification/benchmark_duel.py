"""Measure opt-in duel bookkeeping against the same sandbox simulation.

Uses the shipped models, fixed seed and identical explicit starting positions.
Loading, raster rendering and FPS pacing are excluded. Each alternating-order
pair checks exact final tensor and Torch RNG equality. No host paths or model
fingerprints are included in the public report.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from arena import Arena, parse_args
from clash import list_targets
from runtime import runtime_info


def run_trial(mode, duel, options):
    total = options.warmup + options.steps
    argv = ["--device", "cpu", "--cpu-threads", "1", "--seed", "42", "--mode", mode,
            "--left-pos", "16,23", "--right-pos", "31,23", "--round-ticks", str(total),
            "--warmup-ticks", str(options.warmup), "--grid-size", "48"]
    if duel:
        argv.append("--duel")
    world = Arena(parse_args(argv), list_targets())
    world.step(options.warmup)
    samples = []
    for _ in range(options.steps):
        start = time.perf_counter_ns()
        world.step()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": statistics.fmean(samples),
        "raw_ms": samples,
    }, world, torch.get_rng_state().clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("steps/repeats must be positive; warmup must be nonnegative")
    results = []
    for mode in ("hard", "soft"):
        for repeat in range(args.repeats):
            pair = {}
            for duel in ((False, True) if repeat % 2 == 0 else (True, False)):
                pair[duel] = run_trial(mode, duel, args)
            sandbox, scored = pair[False][1], pair[True][1]
            same = all(torch.equal(a, b) for a, b in zip(
                (sandbox.a, sandbox.b, sandbox.owner, sandbox.control),
                (scored.a, scored.b, scored.owner, scored.control)))
            same_rng = torch.equal(pair[False][2], pair[True][2])
            if not same or not same_rng:
                raise RuntimeError("Duel bookkeeping changed simulation tensors or RNG")
            results.append({"mode": mode, "repeat": repeat + 1,
                            "sandbox": pair[False][0], "duel": pair[True][0],
                            "same_final_tensors": same, "same_torch_rng": same_rng})
    report = {"runtime": runtime_info(seed=42), "grid_size": 48,
              "models": [{"name": bundle["name"], "seed_dir": bundle["seed_dir"]}
                         for bundle in (sandbox.left, sandbox.right)],
              "performance_warmup_ticks": args.warmup, "measured_ticks": args.steps,
              "round_ticks": args.warmup + args.steps, "repeats": args.repeats,
              "scope": "CPU simulation and bookkeeping only; no rendering, pacing, or loading",
              "results": results}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
