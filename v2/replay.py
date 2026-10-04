"""Re-run a completed CPU duel recipe and check its exact integer scores.

Only the recipe and optional output destinations are accepted. No UI, training,
model path, device, simulation or rule override is exposed by this command.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from duel_recipe import (DIAGNOSTIC_KEYS, RecipeValidationError, build_recipe, checkpoint_seed,
                         load_recipe, paths_alias, validate_recipe, validate_runtime)


INTERPRETATION = (
    "A score match means only that both saved integer cell-tick totals match. "
    "It does not verify checkpoint identity, provenance, bitwise state, or balance. "
    "Software or platform changes may change results."
)


class ReplayLoadError(ValueError):
    """The recipe's exact local cultures could not be loaded safely."""


def resolve_recipe_models(recipe, catalog):
    """Resolve stems only against the supplied local catalog; never input paths."""
    recipe = validate_recipe(recipe)
    catalog = list(catalog)
    lookup = {}
    for index, path in enumerate(catalog, 1):
        stem = Path(path).stem
        if stem in lookup:
            raise ReplayLoadError("local target catalog has duplicate stems")
        lookup[stem] = index
    selections = {}
    for side, model in recipe["models"].items():
        if model["target"] not in lookup:
            raise ReplayLoadError(f"target {model['target']} is unavailable in the local catalog")
        selections[side] = lookup[model["target"]]
    return selections


def recipe_to_args(recipe, catalog):
    """Construct strict CPU Arena arguments from validated recipe values."""
    from arena import parse_args as parse_arena_args

    recipe = validate_recipe(recipe)
    selections = resolve_recipe_models(recipe, catalog)
    config, runtime = recipe["configuration"], recipe["runtime"]
    argv = ["--duel", "--device", "cpu", "--bootstrap-steps", "0", "--cpu-threads", str(runtime["cpu_threads"])]
    for key, value in (("mode", config["mode"]), ("seed", config["simulation_seed"]),
                       ("grid-size", config["grid_size"]), ("round-ticks", config["round_ticks"]),
                       ("warmup-ticks", config["warmup_ticks"])):
        argv.extend(("--" + key, str(value)))
    for side, index in selections.items():
        argv.extend(("--" + side, str(index), "--" + side + "-seed", str(recipe["models"][side]["checkpoint_seed"])))
        if config["placement"] == "custom":
            argv.extend(("--" + side + "-pos", ",".join(map(str, config["starting_positions"][side]))))
    for key, value in config["rules"].items():
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    return parse_arena_args(argv)


def runtime_warnings(recipe, runtime):
    """Version/platform differences are diagnostics, never a substitute score."""
    expected = validate_recipe(recipe)["runtime"]
    actual = validate_runtime(runtime, exact=False)
    return [f"Runtime differs: {key} saved={expected[key]!r}, current={actual[key]!r}."
            for key in DIAGNOSTIC_KEYS if expected[key] != actual[key]]


def run_replay(recipe, catalog=None, *, on_warning=None):
    """Run one full round. Return a portable report and its frozen Arena."""
    import torch
    from arena import Arena, list_targets
    from runtime import configure_runtime, runtime_info

    recipe = validate_recipe(recipe)
    catalog = list_targets() if catalog is None else list(catalog)
    args = recipe_to_args(recipe, catalog)
    try:
        configure_runtime(seed=args.seed, cpu_threads=args.cpu_threads,
                          deterministic=recipe["runtime"]["deterministic_algorithms"])
        world = Arena(args, catalog)
        # Arena.reset seeds placement and simulation with its normal deterministic
        # default. Restore only this flag, without reseeding consumed placement RNG.
        torch.use_deterministic_algorithms(recipe["runtime"]["deterministic_algorithms"])
        for side in ("left", "right"):
            bundle, expected = getattr(world, side), recipe["models"][side]
            if (bundle.get("name") != expected["target"] or bundle.get("health") != "ready"
                    or checkpoint_seed(bundle.get("seed_dir")) != expected["checkpoint_seed"]):
                raise ReplayLoadError("loaded checkpoint does not match the exact ready culture and seed")
        if world.size != args.grid_size or world.start_positions != recipe["configuration"]["starting_positions"]:
            raise ReplayLoadError("loaded starting board differs from the recipe")
    except Exception as exc:
        if isinstance(exc, ReplayLoadError):
            raise
        selections = "; ".join(f"{model['target']} [seed_{model['checkpoint_seed']}]"
                               for model in recipe["models"].values())
        raise ReplayLoadError(f"exact local checkpoint loading failed ({type(exc).__name__}): {selections}. "
                              "Provide the matching ready local checkpoints; no substitutes are used.") from exc
    runtime = validate_runtime(runtime_info(), exact=False)
    warnings = runtime_warnings(recipe, runtime)
    if on_warning is not None:
        for warning in warnings:
            on_warning(warning)
    report = {"schema": "petri-clash-duel-replay", "schema_version": 1, "ruleset_version": 1,
              "models": recipe["models"], "configuration": recipe["configuration"],
              "runtime": runtime, "saved_runtime": recipe["runtime"], "warnings": warnings,
              "expected_score": recipe["expected_score"], "observed_score": None,
              "status": "invalid", "result": None, "error": None, "interpretation": INTERPRETATION}
    try:
        world.step(args.round_ticks)
        normal = world.report(0)
        normal["finite"] = bool(all(torch.isfinite(value).all().item()
                                    for value in (world.a, world.b, world.owner, world.control)))
        observed = build_recipe(normal, runtime)
        report["observed_score"] = observed["expected_score"]
        report["status"] = "matched" if observed["expected_score"] == recipe["expected_score"] else "different"
        scores = observed["expected_score"]
        left, right = scores["left_cell_ticks"], scores["right_cell_ticks"]
        ticks = args.round_ticks - args.warmup_ticks
        report["result"] = {"winner": "left" if left > right else "right" if right > left else "draw",
                            "scored_ticks": ticks, "score_unit": "cell-ticks",
                            "metric": "held cells" if args.mode == "hard" else "living cells",
                            "averages": {"left": left / ticks, "right": right / ticks}}
    except Exception as exc:
        # Error text can carry private paths. Keep the file portable and limited
        # to an exception type; an invalid run never receives a matching verdict.
        report["error"] = {"stage": "simulation", "type": type(exc).__name__}
    return report, world


def reject_output_aliases(input_path, *outputs):
    """Reject lexical, symlink and existing hard-link aliases before loading."""
    paths = [Path(input_path), *(Path(path) for path in outputs if path is not None)]
    for index, first in enumerate(paths):
        for second in paths[index + 1:]:
            if paths_alias(first, second):
                raise ValueError("input, report and snapshot must be different files (including aliases)")


def _atomic_output(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(path, report):
    payload = (json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_output(path, lambda stream: stream.write(payload))


def write_snapshot(path, world):
    from PIL import Image

    image = Image.fromarray(world.rgb()).resize((768, 768), Image.Resampling.NEAREST)
    _atomic_output(path, lambda stream: image.save(stream, format="PNG"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("recipe", type=Path, help="saved completed-duel JSON recipe")
    parser.add_argument("--report", type=Path, help="optional portable replay-check JSON")
    parser.add_argument("--snapshot", type=Path, help="optional final board PNG")
    return parser.parse_args(argv)


def print_summary(report, *, include_warnings=True):
    if include_warnings:
        for warning in report["warnings"]:
            print(f"Warning: {warning}", file=sys.stderr)
    for label, key in (("Expected", "expected_score"), ("Observed", "observed_score")):
        score = report[key]
        print(f"{label}: left={score['left_cell_ticks']:,}, right={score['right_cell_ticks']:,} cell-ticks."
              if score is not None else f"{label}: unavailable (invalid simulation).")
    print({"matched": "Saved scores matched.", "different": "Saved scores differed.",
           "invalid": "Simulation invalid; saved scores unverified."}[report["status"]])
    print(INTERPRETATION)


def main(argv=None):
    args = parse_args(argv)
    try:
        reject_output_aliases(args.recipe, args.report, args.snapshot)
        recipe = load_recipe(args.recipe)
        report, world = run_replay(recipe, on_warning=lambda warning: print(f"Warning: {warning}", file=sys.stderr, flush=True))
        if args.report:
            write_report(args.report, report)
        if args.snapshot:
            if report["status"] == "invalid":
                print("replay: snapshot omitted because the simulation is invalid", file=sys.stderr)
            else:
                write_snapshot(args.snapshot, world)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"replay: {exc}", file=sys.stderr)
        return 2
    print_summary(report, include_warnings=False)
    return 0 if report["status"] == "matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
