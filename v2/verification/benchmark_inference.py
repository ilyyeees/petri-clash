"""Compare current CPU inference against the pinned original NCA step.

    python v2/verification/benchmark_inference.py --output inference-efficiency.json

The default run takes a few minutes. It uses actual shipped checkpoints, four
balanced-order timing repeats, eight-tick batches, and four full duel replays.
Loading, warmup, rendering and pacing are excluded from timings. Run without
other CPU-heavy work. This script makes no persistent model or source changes.
"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import numpy as np
import torch

from arena import Arena, parse_args
from clash import list_targets
from nca import NCA
from runtime import runtime_info

BASELINE_REF = "c5209c773f3283556c24ba49248f6b5324cd7bc6"
CASES = (("heart-star", 2, 0), ("heart-flower", 6, 7))


def original_step(ref):
    result = subprocess.run(
        ["git", "show", f"{ref}:v2/nca.py"], cwd=V2_ROOT.parent,
        text=True, capture_output=True, check=True)
    module = types.ModuleType("nca_before_inference_optimization")
    exec(compile(result.stdout, f"<nca@{ref}>", "exec"), module.__dict__)
    return module.NCA.step


def state_tensors(world):
    return tuple(getattr(world, name) for name in ("a", "b", "owner", "control"))


def exact(a, b):
    """Check layout and all bits, including the sign of zero."""
    return (a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
            and torch.equal(a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8)))


def get_rng():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def set_rng(state):
    torch.set_rng_state(state[0])
    random.setstate(state[1])
    np.random.set_state(state[2])


def rng_equality(a, b):
    return {"torch": torch.equal(a[0], b[0]), "python": a[1] == b[1],
            "numpy": (a[2][0] == b[2][0] and np.array_equal(a[2][1], b[2][1])
                      and a[2][2:] == b[2][2:])}


def summarize(samples):
    return {"p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "mean_ms": float(np.mean(samples))}


def make_world(mode, right, seed, options, duel=False):
    argv = ["--device", "cpu", "--cpu-threads", "1", "--grid-size", "48",
            "--left", "1", "--right", str(right), "--seed", str(seed), "--mode", mode]
    if duel:
        argv += ["--duel", "--round-ticks", str(options.replay_ticks),
                 "--warmup-ticks", str(options.replay_warmup)]
    with redirect_stdout(sys.stderr):
        return Arena(parse_args(argv), list_targets())


def model_info(world):
    return [{"name": bundle["name"], "checkpoint_seed": bundle["seed_dir"],
             "channels": bundle["model"].channels,
             "hidden_size": bundle["model"].hidden_size,
             "fire_rate": bundle["model"].fire_rate,
             "dtype": str(next(bundle["model"].parameters()).dtype)}
            for bundle in (world.left, world.right)]


def timed_trial(mode, right, seed, function, options, batch):
    NCA.step = function
    world = make_world(mode, right, seed, options)
    world.step(options.warmup)
    samples = []
    ticks = options.steps if batch == 1 else options.frame_steps
    wall = time.perf_counter()
    cpu = time.process_time()
    for _ in range(ticks // batch):
        start = time.perf_counter_ns()
        world.step(batch)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    wall = time.perf_counter() - wall
    cpu = time.process_time() - cpu
    return {**summarize(samples), "wall_seconds": wall, "cpu_seconds": cpu,
            "cpu_seconds_per_wall_second": cpu / wall,
            "raw_ms": samples}, world, get_rng()


def benchmark(mode, case, baseline, current, options, batch):
    name, right, seed = case
    results = []
    for repeat in range(options.repeats):
        order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
        pair = {}
        for variant in order:
            function = baseline if variant == "baseline" else current
            row, world, rng = timed_trial(mode, right, seed, function, options, batch)
            pair[variant] = world, rng
            row.update(variant=variant, repeat=repeat + 1)
            results.append(row)
        assert all(exact(a, b) for a, b in zip(state_tensors(pair["baseline"][0]),
                                              state_tensors(pair["current"][0])))
        assert all(rng_equality(pair["baseline"][1], pair["current"][1]).values())
        print(f"{name} {mode}: batch {batch}, repeat {repeat + 1} verified", flush=True)
    summary = {}
    for variant in ("baseline", "current"):
        trials = [row for row in results if row["variant"] == variant]
        samples = [value for row in trials for value in row["raw_ms"]]
        cpu = sum(row["cpu_seconds"] for row in trials)
        wall = sum(row["wall_seconds"] for row in trials)
        summary[variant] = {**summarize(samples), "samples": len(samples),
                            "cpu_seconds": cpu, "wall_seconds": wall,
                            "cpu_seconds_per_wall_second": cpu / wall}
    return {"scenario": name, "mode": mode, "seed": seed, "batch_ticks": batch,
            "unit": "one simulation tick" if batch == 1 else "eight simulation ticks",
            "models": model_info(world), "placement": "seeded-random sandbox",
            "starting_positions": world.start_positions, "final_state_bits_and_layout_equal": True,
            "final_rng_equal": {"torch": True, "python": True, "numpy": True},
            "summary": summary, "trials": results}


def verify_replay(mode, case, baseline, current, options):
    name, right, seed = case
    a = make_world(mode, right, seed, options, duel=True)
    b = make_world(mode, right, seed, options, duel=True)
    models = {id(bundle["model"]): bundle["model"]
              for world in (a, b) for bundle in (world.left, world.right)}
    saved = {key: {name: value.clone() for name, value in model.state_dict().items()}
             for key, model in models.items()}
    ra = get_rng()
    rb = ra
    for tick in range(options.replay_ticks):
        NCA.step = baseline
        set_rng(ra)
        advanced_a = a.step()
        ra = get_rng()
        NCA.step = current
        set_rng(rb)
        old = state_tensors(b)
        copies = tuple(value.clone() for value in old)
        advanced_b = b.step()
        rb = get_rng()
        assert advanced_a == advanced_b == 1, (name, mode, tick, "progress")
        assert a.steps == b.steps == tick + 1, (name, mode, tick, "arena tick")
        assert a.duel.snapshot()["tick"] == b.duel.snapshot()["tick"] == tick + 1, (name, mode, tick, "duel tick")
        assert all(exact(x, y) for x, y in zip(old, copies)), (name, mode, tick, "input mutation")
        assert all(exact(x, y) for x, y in zip(state_tensors(a), state_tensors(b))), (name, mode, tick, "state")
        assert all(rng_equality(ra, rb).values()), (name, mode, tick, "RNG")
        assert a.duel.snapshot() == b.duel.snapshot(), (name, mode, tick, "duel")
    assert all(exact(value, saved[key][name]) for key, model in models.items()
               for name, value in model.state_dict().items())
    for world in (a, b):
        terminal = world.duel.snapshot()
        assert terminal["valid"] and terminal["finished"] and terminal["phase"] == "finished", (name, mode, "invalid or unfinished")
        assert terminal["tick"] == terminal["total_ticks"] == options.replay_ticks, (name, mode, "endpoint")
    print(f"{name} {mode}: {options.replay_ticks}-tick exact replay verified", flush=True)
    return {"scenario": name, "mode": mode, "seed": seed, "models": model_info(b),
            "ticks": options.replay_ticks, "warmup_ticks": options.replay_warmup,
            "placement": "mirrored duel", "starting_positions": b.start_positions,
            "one_tick_advanced_each_iteration": True, "finished_valid_at_exact_endpoint": True,
            "every_state_bitwise_equal": True, "every_dtype_and_stride_equal": True,
            "every_input_unchanged": True, "parameters_and_buffers_unchanged": True,
            "every_rng_equal": {"torch": True, "python": True, "numpy": True},
            "every_duel_snapshot_equal": True, "duel": b.duel.snapshot()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--frame-steps", type=int, default=480)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--replay-ticks", type=int, default=600)
    parser.add_argument("--replay-warmup", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if not __debug__:
        parser.error("Run without Python -O; exact verification requires assertions")
    if (options.steps < 1 or options.frame_steps < 8 or options.frame_steps % 8
            or options.warmup < 0 or options.repeats < 1
            or not 0 <= options.replay_warmup < options.replay_ticks):
        parser.error("positive steps/repeats, frame-steps divisible by 8, nonnegative warmup, and 0 <= replay-warmup < replay-ticks required")
    current = NCA.step
    baseline = original_step(options.baseline_ref)
    results = []
    replay = []
    try:
        for case in CASES:
            for mode in ("hard", "soft"):
                results.append(benchmark(mode, case, baseline, current, options, batch=1))
        for mode in ("hard", "soft"):
            results.append(benchmark(mode, CASES[0], baseline, current, options, batch=8))
        for case in CASES:
            for mode in ("hard", "soft"):
                replay.append(verify_replay(mode, case, baseline, current, options))
        report = {
            "schema": "petri-clash-inference-efficiency", "schema_version": 1,
            "baseline_ref": options.baseline_ref, "candidate": "current working-tree NCA.step",
            "runtime": runtime_info(), "grid_size": 48,
            "affinity_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cpu_quota": "not measured by this benchmark",
            "deterministic_fill_uninitialized_memory": bool(torch.utils.deterministic.fill_uninitialized_memory),
            "measurement": {"warmup_ticks": options.warmup, "measured_single_ticks": options.steps,
                            "measured_frame_ticks": options.frame_steps, "repeats": options.repeats,
                            "order": "baseline/current on even zero-based repeats, reversed on odd repeats",
                            "scope": "CPU eager fp32 NCHW sandbox simulation and normal Arena bookkeeping; seeded-random placements",
                            "replay_scope": "separate untimed mirrored duels with full scoring",
                            "excluded": ["loading", "warmup", "rendering", "event handling", "FPS pacing", "replay audit"],
                            "resource_note": "CPU time includes all process threads. No process-RSS improvement is claimed."},
            "results": results, "replay": replay,
            "limits": ["CPU only; GPU not run", "No cross-version or cross-device bitwise guarantee",
                       "Competing host processes and cgroup quota are not measured by this benchmark",
                       "No training, checkpoint writes, deterministic-policy changes, or simulation worker threads"]}
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        NCA.step = current


if __name__ == "__main__":
    main()
