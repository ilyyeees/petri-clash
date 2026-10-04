"""Compare paused board reuse with the original raster path on identical frames.

Uses shipped heart/star weights, one Torch CPU thread and SDL dummy. Every
frame measures the original path twice (baseline/control) and the current
cache, rotating their order. Simulation, warmup, pixel/state checks and native
presentation are outside the timed region. No Git baseline checkout is needed.
"""

import argparse
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
from types import MethodType
from unittest.mock import patch

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import numpy as np
import pygame
import torch

from arena import Arena, ArenaUI, parse_args
from clash import list_targets
from runtime import runtime_info


VARIANTS = ("baseline", "control", "cached")


def uncached_board(ui, array):
    """The original field raster operations, without retained images/surfaces."""
    surface = ui.pg.surfarray.make_surface(array.swapaxes(0, 1))
    return ui.pg.transform.scale(surface, ui.board.size)


def identity(world, ui):
    # Raw identities are compared privately and never written to the report.
    tensors = tuple((str(value.dtype), tuple(value.shape), tuple(value.stride()),
                     value.detach().cpu().contiguous().numpy().tobytes())
                    for value in (world.a, world.b, world.owner, world.control))
    rng = (torch.get_rng_state().numpy().tobytes(),
           pickle.dumps(np.random.get_state()), pickle.dumps(random.getstate()))
    controls = (tuple((tuple(rect), action) for rect, action in ui.buttons),
                tuple((tuple(rect), action) for rect, action in ui.result_buttons))
    return world.steps, tensors, rng, controls


def summary(values):
    return {"p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "samples": len(values)}


def retained_bytes(ui):
    surface, pixels = ui._scaled_board, ui._scaled_board_rgb
    return 0 if surface is None else surface.get_pitch() * surface.get_height() + pixels.nbytes


def trial(width, live, repeat, frames, warmup):
    args = parse_args(["--device", "cpu", "--cpu-threads", "1", "--seed", "42",
                       "--window-size", str(width), "--grid-size", "48", "--left", "1", "--right", "2",
                       "--mode", "hard"])
    world = Arena(args, list_targets())
    ui = ArenaUI(world)
    cached = ui.board_surface
    baseline = MethodType(uncached_board, ui)
    timings = {name: [] for name in VARIANTS}
    try:
        world.step(160)
        ui.paused = not live
        for _ in range(warmup):
            if live:
                world.step()
            ui.draw(dt=1 / 30)
        peak_bytes = 0
        for frame in range(frames):
            if live:
                world.step()
            # Easing advances once; all three timed draws see the same frame.
            ui.board_surface = cached
            ui.draw(dt=1 / 30)
            expected = identity(world, ui)
            expected_pixels = pygame.image.tostring(ui.window, "RGB")
            offset = (repeat + frame) % len(VARIANTS)
            order = VARIANTS[offset:] + VARIANTS[:offset]
            for name in order:
                ui.board_surface = cached if name == "cached" else baseline
                start = time.perf_counter_ns()
                ui.draw(dt=0)
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
                assert identity(world, ui) == expected, (width, live, frame, name, "state/RNG/hitboxes")
                assert pygame.image.tostring(ui.window, "RGB") == expected_pixels, (
                    width, live, frame, name, "pixels")
            peak_bytes = max(peak_bytes, retained_bytes(ui))
            if live:
                assert retained_bytes(ui) == 0, "Live rendering retained a cache"
        paired = {name: [value - base for value, base in zip(timings[name], timings["baseline"])]
                  for name in ("control", "cached")}
        return {"width": width, "window": list(ui.window.get_size()), "board": list(ui.board.size),
                "mode": "live" if live else "paused", "repeat": repeat + 1,
                "models": [{"culture": bundle["name"], "checkpoint_seed": bundle["seed_dir"],
                            "channels": bundle["channels"]} for bundle in (world.left, world.right)],
                "draw": {name: summary(values) for name, values in timings.items()},
                "paired_delta": {name: summary(values) for name, values in paired.items()},
                "retained_raster_bytes_peak": peak_bytes,
                "every_frame_pixels_state_rng_hitboxes_equal": True,
                "samples": timings}
    finally:
        pygame.quit()


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=positive_int, default=120)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if not __debug__:
        parser.error("Run without Python -O; exact verification requires assertions")
    runs = []
    with patch("pygame.mouse.get_pos", return_value=(0, 0)), \
            patch("pygame.mouse.get_pressed", return_value=(False, False, False)):
        for repeat in range(options.repeats):
            scenes = [(width, live) for width in (860, 1100) for live in (False, True)]
            for width, live in scenes[::(-1 if repeat % 2 else 1)]:
                run = trial(width, live, repeat, options.frames, options.warmup)
                runs.append(run)
                print(f"{width}px {run['mode']} repeat {repeat + 1}: "
                      f"original {run['draw']['baseline']['p50_ms']:.3f} ms; "
                      f"cached {run['draw']['cached']['p50_ms']:.3f} ms", flush=True)
    results = []
    for width in (860, 1100):
        for mode in ("paused", "live"):
            group = [run for run in runs if (run["width"], run["mode"]) == (width, mode)]
            samples = {name: [value for run in group for value in run["samples"][name]]
                       for name in VARIANTS}
            results.append({"width": width, "mode": mode,
                            "draw": {name: summary(values) for name, values in samples.items()},
                            "paired_delta": {name: summary([value - base for value, base in
                                zip(samples[name], samples["baseline"])]) for name in ("control", "cached")},
                            "retained_raster_bytes_peak": max(run["retained_raster_bytes_peak"] for run in group)})
    for run in runs:
        del run["samples"]
    report = {
        "schema": "petri-clash-frozen-board-rendering", "schema_version": 1,
        "runtime": {**runtime_info(42), "pygame_version": pygame.version.ver},
        "models": runs[0]["models"],
        "configuration": {"grid_size": 48, "initial_growth_ticks": 160, "seed": 42,
                          "frames_per_trial": options.frames, "warmup_frames": options.warmup,
                          "repeats": options.repeats, "torch_threads": 1},
        "scope": "Same-frame rotating baseline, unchanged control and current paused-only cache. "
                 "Warm, settled paused redraws, not first-pause/edit/easing-miss cost. "
                 "One untimed advance of presentation easing per frame. Live scenes step once before drawing. "
                 "Only draw() is timed; loading, simulation, warmup, identity checks and native display/pacing are excluded.",
        "every_frame_pixels_state_rng_hitboxes_equal": True,
        "limits": ["SDL dummy raster measurements, not native-display FPS", "Competing host processes are not controlled",
                   "Cache memory is surface pitch plus RGB bytes, not process RSS", "No training or checkpoint writes"],
        "summary": results, "runs": runs,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
