"""Small, strict recipes for completed CPU duels; no simulation imports or state.

A recipe identifies local catalog cultures and their checkpoint seeds. It is
not a checkpoint, a recording, or proof of model identity. Export is pure apart
from the explicitly requested file write, and never reads a checkpoint path.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile


SCHEMA = "petri-clash-duel-recipe"
SCHEMA_VERSION = RULESET_VERSION = 1
MAX_FILE_BYTES = 64 * 1024
MAX_SEED = 2**32 - 1
RULE_KEYS = ("pressure_gain", "control_decay", "capture_threshold", "release_threshold", "tie_margin")
RUNTIME_KEYS = ("device", "cpu_threads", "deterministic_algorithms", "python_version",
                "torch_version", "numpy_version", "platform", "machine")
DIAGNOSTIC_KEYS = RUNTIME_KEYS[3:]


class RecipeValidationError(ValueError):
    """The input cannot describe a supported, bounded completed duel."""


def _object(value, name, keys=None):
    if type(value) is not dict:
        raise RecipeValidationError(f"{name} must be an object")
    if keys is not None and set(value) != set(keys):
        raise RecipeValidationError(f"{name} has missing or unknown fields")
    return value


def _integer(value, name, minimum=0, maximum=MAX_SEED):
    if type(value) is not int or not minimum <= value <= maximum:
        raise RecipeValidationError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value, name):
    if type(value) not in (int, float):
        raise RecipeValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise RecipeValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise RecipeValidationError(f"{name} must be a finite number")
    return result


def _choice(value, name, choices):
    if type(value) is not str or value not in choices:
        raise RecipeValidationError(f"{name} must be one of {', '.join(choices)}")
    return value


def _target(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise RecipeValidationError("target must be a safe catalog stem (1..80 letters, digits, '_' or '-')")
    return value


def _diagnostic(value, name):
    # Diagnostic text is informational, never an import, shell argument or path.
    if (type(value) is not str or not 1 <= len(value) <= 128
            or any(not 32 <= ord(char) <= 126 or char in "/\\" for char in value)):
        raise RecipeValidationError(f"{name} must be 1..128 safe printable characters, without paths")
    return value


def validate_runtime(raw, *, exact=True):
    raw = _object(raw, "runtime", RUNTIME_KEYS if exact else None)
    try:
        device = _choice(raw["device"], "runtime.device", ("cpu",))
        threads = _integer(raw["cpu_threads"], "runtime.cpu_threads", 1, 64)
        deterministic = raw["deterministic_algorithms"]
        if type(deterministic) is not bool:
            raise RecipeValidationError("runtime.deterministic_algorithms must be a boolean")
        return {"device": device, "cpu_threads": threads, "deterministic_algorithms": deterministic,
                **{key: _diagnostic(raw[key], f"runtime.{key}") for key in DIAGNOSTIC_KEYS}}
    except KeyError as exc:
        raise RecipeValidationError("runtime has missing fields") from exc


def checkpoint_seed(seed_dir):
    """Parse a loaded bundle's actual seed directory, without accepting paths."""
    if type(seed_dir) is not str or not re.fullmatch(r"seed_[0-9]{1,10}", seed_dir):
        raise RecipeValidationError("loaded checkpoint must have an exact seed_N directory name")
    return _integer(int(seed_dir[5:]), "checkpoint_seed")


def validate_recipe(raw):
    """Return fresh builtin JSON data. Unknown keys and coercion are rejected."""
    raw = _object(raw, "recipe", ("schema", "schema_version", "ruleset_version", "models",
                                 "configuration", "runtime", "expected_score"))
    if raw["schema"] != SCHEMA or type(raw["schema"]) is not str:
        raise RecipeValidationError("unsupported recipe schema")
    for key in ("schema_version", "ruleset_version"):
        if _integer(raw[key], key, 1, 1) != 1:
            raise RecipeValidationError(f"unsupported {key}")
    models = _object(raw["models"], "models", ("left", "right"))
    clean_models = {}
    for side in ("left", "right"):
        model = _object(models[side], f"models.{side}", ("target", "checkpoint_seed"))
        clean_models[side] = {"target": _target(model["target"]),
                              "checkpoint_seed": _integer(model["checkpoint_seed"], "checkpoint_seed")}
    config = _object(raw["configuration"], "configuration", (
        "mode", "simulation_seed", "grid_size", "starting_positions", "placement",
        "round_ticks", "warmup_ticks", "rules"))
    mode = _choice(config["mode"], "mode", ("hard", "soft"))
    seed = _integer(config["simulation_seed"], "simulation_seed")
    size = _integer(config["grid_size"], "grid_size", 8, 128)
    total = _integer(config["round_ticks"], "round_ticks", 1, 10000)
    warmup = _integer(config["warmup_ticks"], "warmup_ticks", 0, total - 1)
    placement = _choice(config["placement"], "placement", ("mirrored", "custom"))
    positions = _object(config["starting_positions"], "starting_positions", ("left", "right"))
    clean_positions = {}
    for side in ("left", "right"):
        pair = positions[side]
        if type(pair) is not list or len(pair) != 2:
            raise RecipeValidationError(f"starting_positions.{side} must be [x, y]")
        # reset_world clamps every explicit coordinate to this range. Accepting
        # a border coordinate here would replay a different starting position.
        clean_positions[side] = [_integer(value, f"{side} coordinate", 2, size - 3) for value in pair]
    if placement == "mirrored":
        center, start = (size - 1) // 2, size // 3
        if clean_positions != {"left": [start, center], "right": [size - 1 - start, center]}:
            raise RecipeValidationError("mirrored positions must match the default duel starts")
    rules = _object(config["rules"], "rules", RULE_KEYS)
    clean_rules = {key: _number(rules[key], key) for key in RULE_KEYS}
    if not 0 <= clean_rules["release_threshold"] < clean_rules["capture_threshold"] <= 1:
        raise RecipeValidationError("require 0 <= release_threshold < capture_threshold <= 1")
    if (clean_rules["pressure_gain"] <= 0 or not 0 <= clean_rules["control_decay"] <= 1
            or not 0 <= clean_rules["tie_margin"] <= 1):
        raise RecipeValidationError("pressure_gain must be positive; control_decay and tie_margin must be in [0,1]")
    scores = _object(raw["expected_score"], "expected_score", ("left_cell_ticks", "right_cell_ticks"))
    bound = size * size * (total - warmup)
    clean_scores = {key: _integer(value, key, 0, bound) for key, value in scores.items()}
    if mode == "hard" and sum(clean_scores.values()) > bound:
        raise RecipeValidationError("hard-mode expected scores cannot exceed the board's scored cell-ticks")
    return {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "ruleset_version": RULESET_VERSION,
            "models": clean_models,
            "configuration": {"mode": mode, "simulation_seed": seed, "grid_size": size,
                              "starting_positions": clean_positions, "placement": placement,
                              "round_ticks": total, "warmup_ticks": warmup, "rules": clean_rules},
            "runtime": validate_runtime(raw["runtime"]), "expected_score": clean_scores}


def build_recipe(report, runtime):
    """Capture live report values only, without retaining paths or mutable data.

    The Arena must check all simulation tensors for finiteness before calling.
    A normal match report remains unchanged; only this new object is sanitized.
    """
    report = _object(report, "report")
    try:
        if not isinstance(report.get("duel"), dict):
            raise RecipeValidationError("only finite, valid, completed CPU duels can be saved")
        duel = report["duel"]
        if (report["finite"] is not True or report["device"] != "cpu"
                or report.get("lesson") is not None or duel["finished"] is not True
                or duel["valid"] is not True or duel["phase"] != "finished"
                or duel.get("invalid_reason") is not None):
            raise RecipeValidationError("only finite, valid, completed CPU duels can be saved")
        models = {}
        for side in ("left", "right"):
            model = _object(report[side], f"report.{side}")
            if model["health"] != "ready":
                raise RecipeValidationError("only ready loaded checkpoints can be saved")
            models[side] = {"target": model["name"], "checkpoint_seed": checkpoint_seed(model["seed_dir"])}
        recipe = validate_recipe({
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "ruleset_version": RULESET_VERSION,
            "models": models, "configuration": {
                "mode": report["mode"], "simulation_seed": report["seed"], "grid_size": report["grid_size"],
                "starting_positions": report["starting_positions"], "placement": report["placement"],
                "round_ticks": duel["total_ticks"], "warmup_ticks": duel["warmup_ticks"], "rules": report["rules"]},
            "runtime": validate_runtime(runtime, exact=False),
            "expected_score": {f"{side}_cell_ticks": duel["scores"][side] for side in ("left", "right")}})
        if _integer(report["cpu_threads"], "report.cpu_threads", 1, 64) != recipe["runtime"]["cpu_threads"]:
            raise RecipeValidationError("report and runtime CPU thread counts must agree")
        config = recipe["configuration"]
        total, warmup = config["round_ticks"], config["warmup_ticks"]
        if (duel["mode"] != config["mode"] or _integer(duel["tick"], "duel.tick") != total
                or _integer(report["step"], "report.step") != total
                or _integer(duel["scored_ticks"], "scored_ticks") != total - warmup):
            raise RecipeValidationError("completed duel evidence is inconsistent with the live report")
        return recipe
    except (KeyError, TypeError) as exc:
        raise RecipeValidationError("report has missing or malformed completed-duel fields") from exc


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RecipeValidationError("duplicate JSON field")
        value[key] = item
    return value


def _reject_constant(value):
    raise RecipeValidationError("non-finite JSON numbers are unsupported")


def load_recipe(path):
    """Read at most 64 KiB, rejecting duplicate fields and nonstandard JSON."""
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_FILE_BYTES + 1)
    if len(payload) > MAX_FILE_BYTES:
        raise RecipeValidationError("recipe exceeds the 64 KiB limit")
    try:
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise RecipeValidationError("invalid recipe JSON") from exc
    return validate_recipe(raw)


def _serialize_recipe(recipe):
    payload = (json.dumps(validate_recipe(recipe), indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > MAX_FILE_BYTES:
        raise RecipeValidationError("recipe exceeds the 64 KiB limit")
    return payload


def _write_temp(directory, payload, *, prefix):
    """Create a no-clobber file; remove it if writing or flushing fails."""
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=directory)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def write_recipe(path, recipe):
    """Atomically replace an explicit output after complete validation/encoding."""
    payload = _serialize_recipe(recipe)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_temp(path.parent, payload, prefix=f".{path.name}.")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_unique_recipe(directory, recipe):
    """Save a distinct UI file with race-safe O_EXCL creation, never overwrite."""
    payload = _serialize_recipe(recipe)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return _write_temp(directory, payload, prefix="duel-")


def paths_alias(first, second):
    """True for the same lexical, resolved-symlink or existing hard-linked file."""
    first, second = Path(first), Path(second)
    return (first.resolve() == second.resolve()
            or (first.exists() and second.exists() and os.path.samefile(first, second)))
