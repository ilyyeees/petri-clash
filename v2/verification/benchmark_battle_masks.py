"""Paired CPU timings and exact replays for the production battle mask pipeline.

    python v2/verification/benchmark_battle_masks.py --output battle-masks.json

Run without other CPU-heavy jobs. Git must contain the pinned baseline commit.
Loading, warmup, profiling and replay audits are excluded from measured timings.
No sources, checkpoints, runtime policy or model layers are changed on disk.
"""
import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import weakref

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import numpy as np
import torch

import arena
import battle
from clash import list_targets
from runtime import runtime_info
from benchmark_inference import exact, get_rng, set_rng, rng_equality, state_tensors, model_info, summarize

BASELINE_REF = "cc2084df711ca07902b5a95552fa95354b10992b"


def original_step(ref):
    result = subprocess.run(["git", "show", f"{ref}:v2/battle.py"], cwd=V2_ROOT.parent,
                            text=True, capture_output=True, check=True)
    module = types.ModuleType("battle_before_mask_fusion")
    exec(compile(result.stdout, f"<battle@{ref}>", "exec"), module.__dict__)
    return module.clash_step


def make_world(size=48, right=2, seed=42, duel=False, ticks=600, warmup=60, mode="hard"):
    argv = ["--device", "cpu", "--cpu-threads", "1", "--grid-size", str(size),
            "--left", "1", "--right", str(right), "--seed", str(seed), "--mode", mode]
    if duel:
        argv += ["--duel", "--round-ticks", str(ticks), "--warmup-ticks", str(warmup)]
    with redirect_stdout(sys.stderr):
        return arena.Arena(arena.parse_args(argv), list_targets())


def trial(function, options, size, right, batch, mode):
    arena.clash_step = function
    world = make_world(size=size, right=right, mode=mode)
    world.step(options.warmup)
    samples = []
    ticks = options.steps if batch == 1 else options.frame_steps
    cpu, wall = time.process_time(), time.perf_counter()
    for _ in range(ticks // batch):
        start = time.perf_counter_ns()
        assert world.step(batch) == batch
        samples.append((time.perf_counter_ns() - start) / 1e6)
    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
    return {**summarize(samples), "raw_ms": samples, "cpu_seconds": cpu,
            "wall_seconds": wall}, world, get_rng()


def benchmark(baseline, current, options, size=48, right=2, batch=1, mode="hard", control=False):
    rows = []
    for repeat in range(options.repeats):
        pair = {}
        for variant in (("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")):
            function = baseline if variant == "baseline" or control else current
            row, world, rng = trial(function, options, size, right, batch, mode)
            rows.append({**row, "repeat": repeat + 1, "variant": variant})
            pair[variant] = world, rng
        assert all(exact(a, b) for a, b in zip(state_tensors(pair["baseline"][0]),
                                               state_tensors(pair["candidate"][0])))
        assert all(rng_equality(pair["baseline"][1], pair["candidate"][1]).values())
        print(f"{mode} size {size} right {right} batch {batch} control {control}: pair {repeat + 1} exact", flush=True)
    summary = {}
    for variant in ("baseline", "candidate"):
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {**summarize([sample for row in selected for sample in row["raw_ms"]]),
                            "cpu_seconds": sum(row["cpu_seconds"] for row in selected),
                            "wall_seconds": sum(row["wall_seconds"] for row in selected)}
    return {"mode": mode, "grid_size": size, "batch_ticks": batch,
            "same_baseline_control": control, "models": model_info(world),
            "seed": 42, "starting_positions": world.start_positions,
            "final_state_bytes_dtype_stride_and_rng_equal": True,
            "summary": summary, "trials": rows}


def replay(baseline, current, options, right, seed, size=48):
    a = make_world(size, right, seed, True, options.replay_ticks, options.replay_warmup)
    b = make_world(size, right, seed, True, options.replay_ticks, options.replay_warmup)
    models = {id(bundle["model"]): bundle["model"] for world in (a, b) for bundle in (world.left, world.right)}
    saved = {key: {name: value.clone() for name, value in model.state_dict().items()} for key, model in models.items()}
    ra = rb = get_rng()
    pixel_checks = 0
    for tick in range(options.replay_ticks):
        arena.clash_step = baseline
        set_rng(ra)
        advanced_a = a.step()
        ra = get_rng()
        arena.clash_step = current
        set_rng(rb)
        old = state_tensors(b)
        copies = tuple(value.clone() for value in old)
        advanced_b = b.step()
        rb = get_rng()
        assert advanced_a == advanced_b == 1, (tick, "progress")
        assert a.steps == b.steps == tick + 1, (tick, "clock")
        assert all(exact(x, y) for x, y in zip(old, copies)), (tick, "input mutation")
        assert all(exact(x, y) for x, y in zip(state_tensors(a), state_tensors(b))), (tick, "state")
        assert all(rng_equality(ra, rb).values()), (tick, "RNG")
        assert a.duel.snapshot() == b.duel.snapshot(), (tick, "score")
        if (tick + 1) % 100 == 0 or tick + 1 == options.replay_ticks:
            assert a.stats() == b.stats(), (tick, "stats")
            for view in ("organisms", "territory", "pressure"):
                for colors in (False, True):
                    assert np.array_equal(a.rgb(colors, view), b.rgb(colors, view)), (tick, "RGB")
                    pixel_checks += 1
    assert all(exact(value, saved[key][name]) for key, model in models.items() for name, value in model.state_dict().items())
    for world in (a, b):
        terminal = world.duel.snapshot()
        assert terminal["valid"] and terminal["finished"] and terminal["phase"] == "finished"
        assert terminal["tick"] == terminal["total_ticks"] == options.replay_ticks
    print(f"{size} right {right} seed {seed}: {options.replay_ticks}-tick exact valid duel", flush=True)
    return {"grid_size": size, "models": model_info(b), "seed": seed,
            "ticks": options.replay_ticks, "every_state_byte_dtype_stride_rng_score_equal": True,
            "all_inputs_parameters_buffers_unchanged": True, "exact_board_rgb_checks": pixel_checks,
            "finished_valid_at_exact_endpoint": True, "duel": b.duel.snapshot()}


def profile_allocations(function, steps):
    arena.clash_step = function
    world = make_world()
    world.step(40)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as trace:
        world.step(steps)
    return {"ticks": steps, "operators": [
        {"name": event.key, "calls_per_tick": event.count / steps,
         "self_cpu_us_per_tick": event.self_cpu_time_total / steps,
         "self_cpu_bytes_per_tick": event.self_cpu_memory_usage / steps}
        for event in trace.key_averages()
        if event.key in ("aten::where", "aten::fill_", "aten::bitwise_and", "aten::empty_strided",
                          "aten::empty", "aten::isfinite", "aten::all", "aten::abs", "aten::amax")],
        "note": "Profiler allocation attribution, not process RSS or peak live memory; timings are instrumented."}


def resource_check(current, ticks):
    arena.clash_step = current
    world = make_world()
    samples, retired = [], []
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else None
    for tick in range(1, ticks + 1):
        old = state_tensors(world)
        retired.extend(weakref.ref(value) for value in old)
        del old
        assert world.step() == 1
        assert all(bool(torch.isfinite(value).all()) for value in state_tensors(world))
        if tick % 500 == 0 or tick == ticks:
            gc.collect()
            assert all(ref() is None for ref in retired), (tick, "retired state still retained")
            retired.clear()
            rss = None
            if page_size and Path("/proc/self/statm").is_file():
                rss = int(Path("/proc/self/statm").read_text().split()[1]) * page_size
            samples.append({"tick": tick, "rss_bytes_after_gc": rss,
                            "cached_models": len(world.cache),
                            "living_cells": [int((value[:, 3:4] > .1).sum()) for value in (world.a, world.b)]})
    return {"ticks": ticks, "every_applied_state_finite": True,
            "all_retired_state_objects_reclaimed": True, "samples": samples,
            "note": "Bounded current-process check after benchmark/profiler; sampled retained RSS after GC, not peak memory or proof of leak freedom."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--frame-steps", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--replay-ticks", type=int, default=600)
    parser.add_argument("--replay-warmup", type=int, default=60)
    parser.add_argument("--resource-ticks", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if not __debug__:
        parser.error("Run without Python -O; invariant assertions must execute")
    if (options.steps < 1 or options.frame_steps < 8 or options.frame_steps % 8
            or options.warmup < 0 or options.repeats < 1 or options.resource_ticks < 1
            or not 0 <= options.replay_warmup < options.replay_ticks):
        parser.error("positive counts, frame-steps divisible by eight, nonnegative warmup and valid replay window required")
    candidate_source = (V2_ROOT / "battle.py").read_bytes()
    current = battle.clash_step
    original = arena.clash_step
    baseline = original_step(options.baseline_ref)
    try:
        results = [benchmark(baseline, current, options, size=size) for size in (32, 48, 64, 96)]
        results.append(benchmark(baseline, current, options, right=6))
        results.append(benchmark(baseline, current, options, batch=8))
        results.append(benchmark(baseline, current, options, control=True))
        # Unchanged soft rules are an additional host/timing negative control.
        results.append(benchmark(baseline, current, options, mode="soft"))
        replays = [replay(baseline, current, options, right, seed, size)
                   for right, seed, size in ((2, 0, 48), (6, 7, 48), (2, 42, 32), (2, 1729, 64))]
        profiles = {name: profile_allocations(function, 20)
                    for name, function in (("baseline", baseline), ("candidate", current))}
        resources = resource_check(current, options.resource_ticks)
        if (V2_ROOT / "battle.py").read_bytes() != candidate_source:
            raise RuntimeError("battle.py changed during measurement; discard this run")
        report = {"schema": "petri-clash-battle-mask-efficiency", "schema_version": 1,
                  "baseline_ref": options.baseline_ref,
                  "candidate": "current production battle.clash_step",
                  "candidate_battle_source_sha256": hashlib.sha256(candidate_source).hexdigest(),
                  "runtime": runtime_info(),
                  "affinity_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  "deterministic_fill_uninitialized_memory": bool(torch.utils.deterministic.fill_uninitialized_memory),
                  "measurement": {"warmup_ticks": options.warmup, "single_ticks_per_trial": options.steps,
                                  "eight_tick_work_per_trial": options.frame_steps, "repeats": options.repeats,
                                  "order": "baseline/candidate, reversing each repeat",
                                  "scope": "Normal Arena stepping with only battle.clash_step switched; identical current NCA and runtime in both variants",
                                  "excluded": ["loading", "warmup", "rendering", "pacing", "profile", "replay", "resource checks"]},
                  "results": results, "replays": replays, "profiles": profiles, "resource_check": resources,
                  "limits": ["CPU eager float32; no native-display FPS or GPU measurement",
                             "Unrelated host load and CPU quota not measured",
                             "No cross-platform or cross-version bitwise guarantee",
                             "Fewer temporary output allocations do not establish lower process RSS"]}
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    finally:
        arena.clash_step = original
    print(f"Verified report written to {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
