"""Small seeded, side-swapped duel comparisons, with no training or rendering.

A is the culture selected by --left; B is selected by --right, even when both
select the same culture. Each seed runs A-left/B-right, then B-left/A-right.
Scores are exact cell-ticks, not a rating or evidence of general balance.
"""

from __future__ import annotations

import argparse
import copy
import json
from numbers import Integral
from pathlib import Path
import sys
import time


ORIENTATIONS = ({"left": "A", "right": "B"}, {"left": "B", "right": "A"})
OPTIONS = ("mode", "left", "right", "left_seed", "right_seed", "grid_size",
           "round_ticks", "warmup_ticks", "device", "cpu_threads")
RULES = ("pressure_gain", "control_decay", "capture_threshold", "release_threshold", "tie_margin")
INTERPRETATION = (
    "A small seeded comparison under the reported settings, not a general ranking "
    "or statistical significance claim. Mirrored starts and side swaps do not "
    "prove fair dynamics. Soft mode counts independent living cells, including overlap."
)


def _integer(value, name, minimum=0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral) or value < minimum
            or (maximum is not None and value > maximum)):
        upper = maximum if maximum is not None else "unbounded"
        raise ValueError(f"{name} must be an integer in [{minimum}, {upper}]")
    return int(value)


def validate_seeds(values):
    seeds = tuple(_integer(value, "seed", maximum=2**32 - 1) for value in values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    return seeds


def parse_seeds(text):
    try:
        parts = [part.strip() for part in text.split(",")]
        if any(not part or not part.isascii() or not part.isdecimal() for part in parts):
            raise ValueError("expected comma-separated nonnegative integer seeds")
        return validate_seeds(tuple(int(part) for part in parts))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _arena_args(args, seed, swapped=False):
    # A deliberately narrow CLI: reuse the Arena's defaults and validation,
    # without exposing its UI, bootstrap, checkpoint-path or research overrides.
    from arena import parse_args as parse_arena_args

    argv = ["--duel", "--seed", str(seed)]
    values = {key: getattr(args, key) for key in OPTIONS}
    if swapped:
        values["left"], values["right"] = values["right"], values["left"]
        values["left_seed"], values["right_seed"] = values["right_seed"], values["left_seed"]
    for key, value in values.items():
        if value is not None:
            argv.extend(("--" + key.replace("_", "-"), str(value)))
    return parse_arena_args(argv)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=int, default=1, help="A: target index (default: 1, heart)")
    parser.add_argument("--right", type=int, default=2, help="B: target index (default: 2, star)")
    parser.add_argument("--left-seed", type=int, help="checkpoint seed for A in both orientations")
    parser.add_argument("--right-seed", type=int, help="checkpoint seed for B in both orientations")
    parser.add_argument("--seeds", type=parse_seeds, default=(0, 7, 42),
                        help="distinct simulation seeds (default: 0,7,42; six rounds)")
    parser.add_argument("--mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--grid-size", type=int, default=0, help="0 uses checkpoint grid size")
    parser.add_argument("--round-ticks", type=int, default=600, help="total ticks including warmup (default: 600)")
    parser.add_argument("--warmup-ticks", type=int, default=60, help="unscored initial ticks (default: 60)")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, help="default: PETRI_CPU_THREADS or 1")
    parser.add_argument("--report", type=Path, help="optional detailed JSON destination")
    requested = parser.parse_args(argv)
    args = _arena_args(requested, requested.seeds[0])
    from runtime import resolve_cpu_threads

    try:
        if min(args.left, args.right) < 1:
            raise ValueError("--left and --right must be positive target indices")
        for seed in (args.left_seed, args.right_seed):
            if seed is not None:
                _integer(seed, "checkpoint seed", maximum=2**32 - 1)
        args.cpu_threads = resolve_cpu_threads(args.cpu_threads)
    except ValueError as exc:
        parser.error(str(exc))
    args.seeds, args.report = requested.seeds, requested.report
    return args


def _outcome(a, b):
    return "A" if a > b else "B" if b > a else "draw"


def _complete_result(leg):
    """Validate terminal scoring evidence before using it in a comparison."""
    result = leg.get("result")
    if leg.get("status") != "complete" or not isinstance(result, dict):
        return False
    try:
        total = _integer(result["total_ticks"], "total_ticks", 1)
        warmup = _integer(result["warmup_ticks"], "warmup_ticks", maximum=total - 1)
        left, right = (_integer(result["scores"][side], side) for side in ("left", "right"))
        return (result["valid"] is True and result["finished"] is True
                and result.get("invalid_reason") is None
                and result["phase"] == "finished" and _integer(result["tick"], "tick") == total
                and _integer(result["scored_ticks"], "scored_ticks") == total - warmup
                and result["mode"] in ("hard", "soft")
                and result["winner"] == ("left" if left > right else "right" if right > left else "draw"))
    except (KeyError, TypeError, ValueError):
        return False


def aggregate_pairs(seeds, legs):
    """Return fresh exact summaries; never mutate or retain caller dictionaries.

    A missing, invalid, failed or inconsistent leg suppresses the overall
    outcome. Completed-pair totals remain explicitly diagnostic in that case.
    """
    seeds = validate_seeds(seeds)
    indexed = {}
    for index, leg in enumerate(legs):
        seed = _integer(leg["seed"], "seed", maximum=2**32 - 1)
        sides = leg["sides"]
        if seed not in seeds or sides not in ORIENTATIONS:
            raise ValueError("leg must use a requested seed and an A/B side mapping")
        key = seed, sides["left"]
        if key in indexed:
            raise ValueError("duplicate seed/orientation leg")
        indexed[key] = index, leg

    pairs, totals, signature = [], {"A": 0, "B": 0}, None
    for seed in seeds:
        rows = [indexed[(seed, label)] for label in ("A", "B") if (seed, label) in indexed]
        valid = all(_complete_result(leg) for _, leg in rows)
        for _, leg in rows:
            if _complete_result(leg):
                result = leg["result"]
                current = tuple(result[key] for key in ("mode", "total_ticks", "warmup_ticks"))
                signature = signature or current
                valid = valid and current == signature
        status = "invalid" if not valid else "complete" if len(rows) == 2 else "incomplete"
        scores, winners = {"A": 0, "B": 0}, []
        if status == "complete":
            for _, leg in rows:
                result, sides = leg["result"], leg["sides"]
                for side, label in sides.items():
                    scores[label] += int(result["scores"][side])
                winners.append(sides.get(result["winner"], "draw"))
            for label in totals:
                totals[label] += scores[label]
        pairs.append({"seed": seed, "status": status, "leg_indices": [index for index, _ in rows],
                      "scores": scores if status == "complete" else None,
                      "paired_outcome": _outcome(scores["A"], scores["B"])
                      if status == "complete" else None,
                      "leg_winners": winners if status == "complete" else None,
                      "winner_changes_after_swap": winners[0] != winners[1]
                      if status == "complete" else None})
    complete = all(pair["status"] == "complete" for pair in pairs)
    return {"status": "complete" if complete else "invalid" if any(
                pair["status"] == "invalid" for pair in pairs) else "incomplete",
            "expected_pairs": len(seeds), "complete_pairs": sum(pair["status"] == "complete" for pair in pairs),
            "scores": totals.copy() if complete else None, "completed_pair_scores": totals.copy(),
            "paired_outcome": _outcome(totals["A"], totals["B"]) if complete else None,
            "per_seed": pairs}


def run_comparison(args, progress=None):
    """Run fresh Arena duels, stopping immediately after the first bad leg."""
    from arena import Arena, list_targets
    from runtime import configure_runtime
    from trainer.common import pick_device

    seeds = validate_seeds(args.seeds)
    base = _arena_args(args, seeds[0])
    targets = list_targets()
    if not targets or not all(1 <= value <= len(targets) for value in (base.left, base.right)):
        raise ValueError(f"--left and --right must be between 1 and {len(targets)}")
    base.device = pick_device(base.device)
    runtime = configure_runtime(seed=seeds[0], cpu_threads=base.cpu_threads)
    runtime["device"], runtime["seed"] = base.device, None
    models = {label: {"target_index": index, "name": targets[index - 1].stem,
                      "requested_seed": seed, "seed_dir": None}
              for label, index, seed in (("A", base.left, base.left_seed), ("B", base.right, base.right_seed))}
    report = {"schema": "petri-clash-paired-duel", "schema_version": 1, "models": models,
              "runtime": runtime, "configuration": {
                  "mode": base.mode, "seeds": list(seeds), "round_ticks": base.round_ticks,
                  "warmup_ticks": base.warmup_ticks, "requested_grid_size": base.grid_size,
                  "grid_size": None, "placement": "mirrored", "starting_positions": None,
                  "rules": {key: getattr(base, key) for key in RULES},
                  "score_unit": "cell-ticks", "metric": "held cells" if base.mode == "hard" else "living cells"},
              "interpretation": INTERPRETATION, "legs": []}
    started = time.perf_counter()
    for seed in seeds:
        for swapped, sides in enumerate(ORIENTATIONS):
            if progress:
                progress(f"Round {len(report['legs']) + 1}/{2 * len(seeds)}: seed {seed}, "
                         f"{sides['left']}-left/{sides['right']}-right")
            leg = {"seed": seed, "sides": sides.copy(), "status": "failed", "result": None,
                   "starting_positions": None, "models": None, "error": None}
            world, stage = None, "model loading or round setup"
            leg_started = time.perf_counter()
            try:
                world = Arena(_arena_args(base, seed, bool(swapped)), targets)
                leg["starting_positions"] = copy.deepcopy(world.start_positions)
                leg["models"] = {sides[side]: {"name": bundle["name"],
                                    "seed_dir": Path(bundle["seed_dir"]).name}
                                 for side, bundle in (("left", world.left), ("right", world.right))}
                for label, identity in leg["models"].items():
                    if identity["name"] != models[label]["name"] or (
                            models[label]["seed_dir"] is not None and identity["seed_dir"] != models[label]["seed_dir"]):
                        raise ValueError("selected model identity changed between legs")
                    models[label]["seed_dir"] = identity["seed_dir"]
                config = report["configuration"]
                if config["grid_size"] is None:
                    config["grid_size"] = world.size
                    config["starting_positions"] = copy.deepcopy(world.start_positions)
                if world.size != config["grid_size"] or world.start_positions != config["starting_positions"]:
                    raise ValueError("grid or starting positions changed between legs")
                stage = "simulation"
                world.step(base.round_ticks)
                leg["result"] = world.duel.snapshot()
                leg["status"] = "complete" if leg["result"]["valid"] and world.duel.finished else "invalid"
                if not _complete_result(leg):
                    leg["status"] = "invalid"
            except Exception as exc:
                # Exception text can contain absolute checkpoint/host paths.
                # Keep portable diagnostics in JSON, full details in the terminal.
                leg["error"] = {"stage": stage, "type": type(exc).__name__}
                if world is not None:
                    leg["result"] = world.duel.snapshot()
                if progress:
                    progress(f"Failed during {stage}: {type(exc).__name__}: {exc}")
            leg["elapsed_seconds"] = time.perf_counter() - leg_started
            report["legs"].append(leg)
            if leg["status"] != "complete":
                break
        if report["legs"][-1]["status"] != "complete":
            break
    report["summary"] = aggregate_pairs(seeds, report["legs"])
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def print_summary(report):
    summary = report["summary"]
    outcome = summary["paired_outcome"]
    print("; ".join(f"{label}: {model['name']} [{model['seed_dir'] or 'unresolved'}]"
                    for label, model in report["models"].items()))
    if outcome is None:
        print(f"Comparison {summary['status']}: {summary['complete_pairs']}/{summary['expected_pairs']} "
              "complete pairs; no comparison winner. Stopped at the first invalid or failed leg.")
    else:
        scores = summary["scores"]
        verdict = "Paired score is an exact draw" if outcome == "draw" else f"Paired score favors {outcome}"
        print(f"{verdict}: A={scores['A']:,}, B={scores['B']:,} cell-ticks.")
    for pair in summary["per_seed"]:
        if pair["status"] == "complete":
            print(f"  Seed {pair['seed']}: {pair['paired_outcome']}; "
                  f"A={pair['scores']['A']:,}, B={pair['scores']['B']:,}; "
                  f"leg winners {' / '.join(pair['leg_winners'])}; "
                  f"winner changed after swap: {pair['winner_changes_after_swap']}")
    print(INTERPRETATION)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run_comparison(args, progress=lambda message: print(message, file=sys.stderr, flush=True))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"compare: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0 if report["summary"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
