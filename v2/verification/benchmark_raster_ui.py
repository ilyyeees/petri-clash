"""Measure exact-pixel raster savings against the pre-optimization arena.

    python v2/verification/benchmark_raster_ui.py --output raster-efficiency.json

Uses actual shipped heart/star checkpoints, one CPU thread, and SDL dummy.
Loading, warmup, native display/compositor, event handling and FPS pacing are
excluded. Requires the baseline Git revision locally; no training is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import types

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import numpy as np
import pygame
import torch

import arena as current
from clash import list_targets

BASELINE_COMMIT = "a31d6c84ae3cd3d77f7436e54b4d1e817f5f14f5"


def baseline_module():
    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:v2/arena.py"], cwd=V2_ROOT.parent,
        text=True, capture_output=True, check=True).stdout
    module = types.ModuleType("arena_before_raster_optimization")
    module.__file__ = str(V2_ROOT / "arena.py")
    exec(compile(source, f"<arena@{BASELINE_COMMIT}>", "exec"), module.__dict__)
    return module


def summarize(values):
    return {"p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(np.mean(values))}


def digest(value):
    return hashlib.sha256(value).digest()


def identities(world, ui):
    # Compare privately; raw state/RNG/screen fingerprints are not exported.
    return {"simulation": tuple(digest(value.detach().cpu().contiguous().numpy().tobytes())
                               for value in (world.a, world.b, world.owner, world.control)),
            "rng": (digest(torch.get_rng_state().numpy().tobytes()),
                    digest(pickle.dumps(np.random.get_state())), digest(pickle.dumps(random.getstate()))),
            "screen": digest(pygame.image.tostring(ui.window, "RGB"))}


def run_trial(module, version, width, speed, repeat, frames):
    args = module.parse_args(["--device", "cpu", "--cpu-threads", "1", "--seed", "42",
                              "--window-size", str(width), "--steps-per-frame", str(speed)])
    world = module.Arena(args, list_targets())
    ui = module.ArenaUI(world)
    timings = {name: [] for name in ("empty_effects", "sidebar", "simulation", "draw", "total")}
    try:
        world.step(20)
        for _ in range(10):
            ui.draw(dt=1 / 30)
        for name, function in (("empty_effects", ui.draw_effects),
                               ("sidebar", lambda: ui.draw_sidebar(ui.rail_x, 308, 286))):
            for _ in range(300):
                ui.buttons = []
                start = time.perf_counter_ns()
                function()
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
        for _ in range(frames):
            start = time.perf_counter_ns()
            world.step(speed)
            middle = time.perf_counter_ns()
            ui.draw(dt=1 / 30)
            end = time.perf_counter_ns()
            timings["simulation"].append((middle - start) / 1e6)
            timings["draw"].append((end - middle) / 1e6)
            timings["total"].append((end - start) / 1e6)
        return {"version": version, "width": width, "speed": speed, "repeat": repeat,
                "window": list(ui.window.get_size()), "board": list(ui.board.size),
                "grid_size": world.size, "steps": world.steps,
                "model_sources": [str(Path(bundle["source"]).relative_to(V2_ROOT))
                                  for bundle in (world.left, world.right)],
                "identities": identities(world, ui), "samples": timings,
                **{name: summarize(values) for name, values in timings.items()}}
    finally:
        pygame.quit()


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=positive_int, default=30)
    parser.add_argument("--repeats", type=positive_int, default=4)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(argv)
    before, runs = baseline_module(), []
    for repeat in range(options.repeats):
        for width in ((1100, 860) if repeat % 2 == 0 else (860, 1100)):
            for speed in ((1, 8) if repeat % 2 == 0 else (8, 1)):
                versions = ((before, "before"), (current, "after"))
                for module, version in (versions if repeat % 2 == 0 else tuple(reversed(versions))):
                    run = run_trial(module, version, width, speed, repeat, options.frames)
                    runs.append(run)
                    print(f"{version} {width}px {speed}x repeat {repeat + 1}: "
                          f"draw p50 {run['draw']['p50_ms']:.3f} ms", file=sys.stderr)
    equal = {name: True for name in ("simulation", "rng", "screen")}
    summaries = []
    for width in (1100, 860):
        for speed in (1, 8):
            group = [run for run in runs if run["width"] == width and run["speed"] == speed]
            for name in equal:
                equal[name] &= all(run["identities"][name] == group[0]["identities"][name] for run in group)
            for version in ("before", "after"):
                subset = [run for run in group if run["version"] == version]
                summaries.append({"width": width, "speed": speed, "version": version,
                                  **{name: summarize([value for run in subset for value in run["samples"][name]])
                                     for name in subset[0]["samples"]}})
    for run in runs:
        del run["identities"], run["samples"]
    report = {"schema_version": 1, "benchmark": "petri-clash-raster-ui-sdl-dummy",
              "baseline_commit": BASELINE_COMMIT, "synthetic_models": False,
              "scope": "Shipped heart/star checkpoints on CPU with one Torch thread. SDL dummy raster only; "
                       "no native display, compositor, vsync, event handling or FPS pacing. "
                       "No action effects are active. Both revisions draw the exact same scene. "
                       "Version, size and speed order alternate. Samples exclude loading and warmup.",
              "configuration": {"frames_per_trial": options.frames, "repeats": options.repeats,
                                "warmup_steps": 20, "draw_warmup_frames": 10, "micro_samples": 300,
                                "window_widths": [1100, 860], "speeds": [1, 8], "seed": 42},
              "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                           "pygame": pygame.version.ver, "numpy": np.__version__},
              "simulation_tensors_identical": equal["simulation"], "rng_states_identical": equal["rng"],
              "final_screen_pixels_identical": equal["screen"], "summary": summaries, "runs": runs}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded)
    print(encoded, end="")
    if not all(equal.values()):
        raise SystemExit("Simulation, RNG, or final screen changed between revisions.")


if __name__ == "__main__":
    main()
