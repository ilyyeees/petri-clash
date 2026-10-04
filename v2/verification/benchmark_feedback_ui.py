"""Compare unpaced SDL-dummy UI cost against the fixed pre-feedback revision.

Run from any directory with the v2 dependencies installed:
    python v2/verification/benchmark_feedback_ui.py --output feedback-performance.json

Uses the actual shipped heart/star checkpoints, not synthetic models. Requires
the baseline revision in local Git history. It does not create a native window
or measure the compositor, vsync, real display latency, or FPS pacing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

V2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V2_ROOT.parent
BASELINE_COMMIT = "ab6e2bd061ee704147bbf9e628f4eb047e28ea7b"
sys.path.insert(0, str(V2_ROOT))

import numpy as np
import pygame
import torch

import arena as current
from clash import list_targets


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def baseline_module():
    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:v2/arena.py"],
        cwd=REPO_ROOT, text=True, capture_output=True, check=True,
    ).stdout
    module = types.ModuleType("arena_before_feedback")
    module.__file__ = str(V2_ROOT / "arena.py")
    exec(compile(source, f"<arena@{BASELINE_COMMIT}>", "exec"), module.__dict__)
    return module


def state_hashes(world):
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in (("left", world.a), ("right", world.b),
                            ("ownership", world.owner), ("control", world.control))
    }


def summarize(values):
    return {"p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(np.mean(values))}


def run_trial(module, version, speed, repeat, options):
    args = module.parse_args(["--device", "cpu", "--cpu-threads", "1",
                              "--steps-per-frame", str(speed),
                              "--window-size", str(options.window_size)])
    world = module.Arena(args, list_targets())
    ui = module.ArenaUI(world)
    timings = {"simulation_per_frame": [], "draw": [], "total_unpaced_frame": []}

    def draw_frame():
        if version == "before":
            # Match the original run(): exact stats were refreshed every frame.
            ui.stats_cache = world.stats()
            ui.draw()
        else:
            ui.draw(dt=1 / 30)

    try:
        world.step(options.warmup)
        for _ in range(options.draw_warmup):
            draw_frame()
        for _ in range(options.frames):
            started = time.perf_counter_ns()
            world.step(speed)
            boundary = time.perf_counter_ns()
            draw_frame()
            ended = time.perf_counter_ns()
            timings["simulation_per_frame"].append((boundary - started) / 1e6)
            timings["draw"].append((ended - boundary) / 1e6)
            timings["total_unpaced_frame"].append((ended - started) / 1e6)
        return {"version": version, "speed": speed, "repeat": repeat,
                "window": list(ui.window.get_size()), "board": list(ui.board.size),
                "grid_size": world.size, "cpu_threads": torch.get_num_threads(),
                "model_sources": [str(Path(bundle["source"]).relative_to(V2_ROOT)) for bundle in (world.left, world.right)],
                **{name: summarize(values) for name, values in timings.items()},
                "state_hashes": state_hashes(world), "raw_ms": timings}
    finally:
        pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=positive_int, default=60)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--draw-warmup", type=positive_int, default=10)
    parser.add_argument("--window-size", type=positive_int, default=1100)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(argv)
    before = baseline_module()
    runs = []
    for repeat in range(options.repeats):
        for speed in (1, 8):
            versions = ((before, "before"), (current, "after"))
            if repeat % 2:
                versions = tuple(reversed(versions))
            for module, version in versions:
                result = run_trial(module, version, speed, repeat, options)
                runs.append(result)
                print(f"{version} {speed}x repeat {repeat + 1}: "
                      f"draw p50 {result['draw']['p50_ms']:.3f} ms", file=sys.stderr)

    summaries, hashes_identical = [], True
    for speed in (1, 8):
        group = [run for run in runs if run["speed"] == speed]
        hashes_identical &= all(run["state_hashes"] == group[0]["state_hashes"] for run in group)
        for version in ("before", "after"):
            subset = [run for run in group if run["version"] == version]
            summaries.append({"version": version, "speed": speed, **{
                name: summarize([sample for run in subset for sample in run["raw_ms"][name]])
                for name in ("simulation_per_frame", "draw", "total_unpaced_frame")}})

    # Compare privately, then omit environment-specific paths and raw fingerprints.
    for run in runs:
        run.pop("state_hashes")
    report = {
        "schema_version": 1,
        "benchmark": "petri-clash-feedback-ui-sdl-dummy",
        "baseline_commit": BASELINE_COMMIT,
        "synthetic_models": False,
        "scope": "Actual shipped heart/star checkpoints on CPU with one Torch thread. "
                 "SDL dummy performs raster drawing and flip calls; no native display, "
                 "compositor, vsync, event handling or FPS pacing is measured. "
                 "Baseline refreshes exact stats every frame as in its run(); current UI "
                 "refreshes them inside draw(). No action effects are active. Layouts "
                 "use the same window dimensions but may use different board sizes. "
                 "Model loading and warmup are excluded; trial order alternates.",
        "configuration": {"frames": options.frames, "repeats": options.repeats,
                          "warmup_steps": options.warmup, "draw_warmup_frames": options.draw_warmup,
                          "window_width": options.window_size, "speeds": [1, 8]},
        "simulation_hashes_identical": bool(hashes_identical),
        "summary": summaries, "runs": runs,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded)
    print(encoded, end="")
    if not hashes_identical:
        raise SystemExit("Simulation hashes changed across UI versions or repeats.")


if __name__ == "__main__":
    main()
