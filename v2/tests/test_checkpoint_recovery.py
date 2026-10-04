"""Malformed local checkpoint recovery, using only tiny temporary fixtures.

The real restricted loader, model/config validation, picker and lesson lifecycle
are exercised on CPU with dummy SDL. No shipped weights are edited or trained.
"""

import copy
import json
import os
from pathlib import Path
import pickle
import random
import sys
from unittest.mock import Mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import numpy as np
from PIL import Image
import pygame
import pytest
import torch

import arena
import checkpoints
import clash
from nca import NCA


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(expected):
    python, numpy, tensor = rng_state()
    assert python == expected[0]
    assert numpy[0] == expected[1][0]
    np.testing.assert_array_equal(numpy[1], expected[1][1])
    assert numpy[2:] == expected[1][2:]
    assert torch.equal(tensor, expected[2])


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    previous_rng = rng_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    torch.set_num_threads(1)
    random.seed(173)
    np.random.seed(173)
    torch.manual_seed(173)
    yield
    pygame.quit()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    random.setstate(previous_rng[0])
    np.random.set_state(previous_rng[1])
    torch.set_rng_state(previous_rng[2])


def small_blob(channels=6, hidden_size=8, fire_rate=.5, grid_size=16):
    return {
        "model": NCA(channels, hidden_size, fire_rate).state_dict(),
        "config": {
            "model": {"channels": channels, "hidden_size": hidden_size, "fire_rate": fire_rate},
            "data": {"grid_size": grid_size},
            "train": {"channels_last": False},
        },
        "score": .01,
    }


def write_blob(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    return path


def assert_clear_error(exc, *terms):
    text = str(exc).lower()
    assert text and len(text) <= 1024
    assert any(term.lower() in text for term in terms), text


class UnsupportedCheckpointValue:
    """An inert unsupported type, never an executable pickle payload."""


@pytest.mark.parametrize("payload", [b"", b"not a checkpoint", b"\x80"],
                         ids=["empty", "invalid-pickle", "incomplete-opcode"])
def test_malformed_bytes_become_clear_value_errors(tmp_path, monkeypatch, payload):
    path = tmp_path / "broken.pt"
    path.write_bytes(payload)
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(checkpoints.torch, "load", loader)
    with pytest.raises(ValueError) as error:
        checkpoints.load_checkpoint_file(path)
    assert isinstance(error.value.__cause__, (EOFError, pickle.UnpicklingError, IndexError))
    assert_clear_error(error.value, "checkpoint")
    assert path.name in str(error.value)
    assert loader.call_count == 1
    assert loader.call_args.kwargs["weights_only"] is True


@pytest.mark.parametrize("legacy_numpy", [False, True])
def test_unsupported_values_never_enable_unrestricted_loading(tmp_path, monkeypatch, legacy_numpy):
    blob = small_blob()
    if legacy_numpy:
        # Insertion order makes the first restricted failure NumPy-specific;
        # the inert unsupported object is rejected on the allowlisted retry.
        blob["rng"] = {"numpy": np.random.get_state()}
    blob["unsupported"] = UnsupportedCheckpointValue()
    path = write_blob(tmp_path / "unsupported.pt", blob)
    previous_globals = set(torch.serialization.get_safe_globals())
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(checkpoints.torch, "load", loader)
    with pytest.raises(ValueError) as error:
        checkpoints.load_checkpoint_file(path)
    assert isinstance(error.value.__cause__, pickle.UnpicklingError)
    assert_clear_error(error.value, "checkpoint")
    assert loader.call_count == (2 if legacy_numpy else 1)
    assert all(call.kwargs["weights_only"] is True for call in loader.call_args_list)
    assert set(torch.serialization.get_safe_globals()) == previous_globals


def test_legacy_numpy_allowlist_still_loads_without_leaking_globals(tmp_path, monkeypatch):
    blob = small_blob()
    blob["rng"] = {"numpy": np.random.get_state()}
    path = write_blob(tmp_path / "legacy.pt", blob)
    previous_globals = set(torch.serialization.get_safe_globals())
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(checkpoints.torch, "load", loader)
    loaded = checkpoints.load_checkpoint_file(path)
    np.testing.assert_array_equal(loaded["rng"]["numpy"][1], blob["rng"]["numpy"][1])
    assert loader.call_count == 2
    assert all(call.kwargs["weights_only"] is True for call in loader.call_args_list)
    assert set(torch.serialization.get_safe_globals()) == previous_globals


@pytest.mark.parametrize("legacy_retry", [False, True])
@pytest.mark.parametrize("failure", [EOFError, IndexError, pickle.UnpicklingError])
def test_expected_deserialization_failures_are_normalized_on_either_attempt(
        tmp_path, monkeypatch, legacy_retry, failure):
    cause = failure("fixture failure")
    attempts = [pickle.UnpicklingError("unsupported numpy RNG")] if legacy_retry else []
    loader = Mock(side_effect=[*attempts, cause])
    monkeypatch.setattr(checkpoints.torch, "load", loader)
    with pytest.raises(ValueError) as error:
        checkpoints.load_checkpoint_file(tmp_path / "bad.pt")
    assert error.value.__cause__ is cause
    assert loader.call_count == (2 if legacy_retry else 1)
    assert all(call.kwargs["weights_only"] is True for call in loader.call_args_list)


@pytest.mark.parametrize("legacy_retry", [False, True])
@pytest.mark.parametrize("failure", [MemoryError, OSError, RuntimeError, AssertionError])
def test_loader_does_not_swallow_resource_or_programming_errors(
        tmp_path, monkeypatch, legacy_retry, failure):
    cause = failure("unrelated failure")
    attempts = [pickle.UnpicklingError("unsupported numpy RNG")] if legacy_retry else []
    loader = Mock(side_effect=[*attempts, cause])
    monkeypatch.setattr(checkpoints.torch, "load", loader)
    with pytest.raises(failure) as error:
        checkpoints.load_checkpoint_file(tmp_path / "bad.pt")
    assert error.value is cause
    assert all(call.kwargs["weights_only"] is True for call in loader.call_args_list)


@pytest.mark.parametrize("blob", [None, [], {}, {"model": None}, {"model": []}])
def test_restricted_but_non_checkpoint_payload_is_rejected(tmp_path, blob):
    path = write_blob(tmp_path / "not-a-model.pt", blob)
    with pytest.raises(ValueError) as error:
        checkpoints.load_checkpoint_file(path)
    assert_clear_error(error.value, "checkpoint", "model")


BAD_SECTIONS = [(section, value) for section in ("model", "data", "train")
                for value in (None, [], 7, "invalid")]
BAD_FIELDS = [
    ("model", "channels", value) for value in (None, [], {}, True, 0, 4, -1, 6.5, "6.5", float("inf"))
] + [
    ("model", "hidden_size", value) for value in (None, [], True, 0, -1, 8.5, float("nan"))
] + [
    ("model", "fire_rate", value) for value in (None, [], True, -.1, 1.1, float("nan"), float("inf"), -float("inf"))
] + [
    ("data", "grid_size", value) for value in (None, [], True, 0, 7, -1, 16.5, float("inf"))
]


@pytest.mark.parametrize("section,value", BAD_SECTIONS)
def test_malformed_config_sections_reject_before_constructing_model(tmp_path, monkeypatch, section, value):
    blob = small_blob()
    blob["config"][section] = value
    path = write_blob(tmp_path / "bad-config.pt", blob)
    constructor = Mock(side_effect=AssertionError("invalid config allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, section)
    constructor.assert_not_called()


@pytest.mark.parametrize("section,field,value", BAD_FIELDS)
def test_invalid_numeric_config_rejects_before_model_allocation(tmp_path, monkeypatch, section, field, value):
    blob = small_blob()
    blob["config"][section][field] = value
    path = write_blob(tmp_path / "bad-number.pt", blob)
    constructor = Mock(side_effect=AssertionError("invalid config allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, field)
    constructor.assert_not_called()


@pytest.mark.parametrize("section,field", [("model", "channels"), ("model", "hidden_size"),
                                            ("model", "fire_rate"), ("data", "grid_size")])
def test_missing_required_config_fields_have_actionable_errors(tmp_path, monkeypatch, section, field):
    blob = small_blob()
    del blob["config"][section][field]
    path = write_blob(tmp_path / "missing-field.pt", blob)
    constructor = Mock(side_effect=AssertionError("incomplete config allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, field)
    constructor.assert_not_called()


@pytest.mark.parametrize("field", ["channels", "hidden_size"])
def test_large_inconsistent_architecture_is_rejected_without_allocation(tmp_path, monkeypatch, field):
    blob = small_blob()
    blob["config"]["model"][field] = 10**9
    path = write_blob(tmp_path / "inconsistent.pt", blob)
    constructor = Mock(side_effect=AssertionError("inconsistent metadata allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, "shape", "state", "parameter", "architecture")
    constructor.assert_not_called()


@pytest.mark.parametrize("key", ["perception_filters", "fc0.weight", "fc0.bias", "fc1.weight", "fc1.bias"])
@pytest.mark.parametrize("kind", ["missing", "wrong-shape", "not-a-tensor"])
def test_incompatible_state_dict_is_rejected_before_model_allocation(tmp_path, monkeypatch, key, kind):
    blob = small_blob()
    if kind == "missing":
        del blob["model"][key]
    else:
        blob["model"][key] = torch.zeros(1) if kind == "wrong-shape" else None
    path = write_blob(tmp_path / "bad-state.pt", blob)
    constructor = Mock(side_effect=AssertionError("invalid weights allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, key, "state", "parameter")
    constructor.assert_not_called()


@pytest.mark.parametrize("metadata", [[], 7, {"": None}, {"": []}, {0: {"version": 1}}])
def test_invalid_state_dict_metadata_is_rejected_before_allocation(tmp_path, monkeypatch, metadata):
    blob = small_blob()
    blob["model"]._metadata = metadata
    path = write_blob(tmp_path / "bad-module-metadata.pt", blob)
    constructor = Mock(side_effect=AssertionError("invalid metadata allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, "metadata", "state")
    constructor.assert_not_called()


@pytest.mark.parametrize("location", ["saved", "companion"])
@pytest.mark.parametrize("config", [None, [], 7, "invalid", {}, {"data": {"grid_size": 16}},
                                    {"model": {"channels": 6, "hidden_size": 8, "fire_rate": .5}}])
def test_invalid_top_level_or_incomplete_config_is_rejected(tmp_path, monkeypatch, location, config):
    blob = small_blob()
    path = tmp_path / "checkpoints" / "best.pt"
    if location == "saved":
        blob["config"] = config
    else:
        del blob["config"]
        (tmp_path / "resolved_config.json").write_text(json.dumps(config))
    write_blob(path, blob)
    constructor = Mock(side_effect=AssertionError("invalid config allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, "config")
    constructor.assert_not_called()


def test_long_invalid_numeric_value_has_a_bounded_field_diagnostic(tmp_path):
    blob = small_blob()
    blob["config"]["model"]["fire_rate"] = "invalid" * 1000
    path = write_blob(tmp_path / "long-number.pt", blob)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, "fire_rate")
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("grid_size", [2**63, str(2**63), 2**32, 2**30],
                         ids=["dimension-overflow", "string-dimension-overflow", "element-overflow", "byte-overflow"])
def test_unrepresentable_grid_is_rejected_before_any_model_allocation(tmp_path, monkeypatch, grid_size):
    blob = small_blob()
    blob["config"]["data"]["grid_size"] = grid_size
    path = write_blob(tmp_path / "unrepresentable-grid.pt", blob)
    # Intercept the first model construction, so a missing guard fails safely.
    # No test attempts to allocate a huge but representable tensor.
    constructor = Mock(side_effect=AssertionError("unsupported grid allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, "grid", "dimension", "size")
    constructor.assert_not_called()


@pytest.mark.parametrize("representation", ["native", "numeric-strings", "integral-floats"])
@pytest.mark.parametrize("fire_rate", [0.0, 1.0])
def test_valid_small_boundary_models_and_omitted_train_still_load(tmp_path, representation, fire_rate):
    blob = small_blob(channels=5, hidden_size=1, fire_rate=fire_rate, grid_size=8)
    del blob["config"]["train"]
    if representation != "native":
        convert = str if representation == "numeric-strings" else float
        for section in ("model", "data"):
            blob["config"][section] = {key: convert(value) for key, value in blob["config"][section].items()}
    path = write_blob(tmp_path / "valid.pt", blob)
    bundle = clash.load_v2_model(path, "cpu")
    assert (bundle["channels"], bundle["grid_size"], bundle["channels_last"]) == (5, 8, False)
    assert bundle["model"].fire_rate == fire_rate
    assert not bundle["model"].training
    states = clash.reset_world(8, "cpu", bundle, bundle)
    assert all(bool(torch.isfinite(value).all()) for value in states)


BAD_METADATA = [None, [], 7, {"stop": None}, {"stop": []}, {"stop": 7},
                {"stop": {"collapsed_score": None}}, {"stop": {"collapsed_score": []}},
                {"stop": {"collapsed_score": -1}}, {"stop": {"collapsed_score": float("nan")}}]


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(clash, "V2_ROOT", tmp_path)
    targets = []
    for stem in ("01_alpha", "02_beta", "03_broken", "04_spare"):
        target = tmp_path / "targets" / f"{stem}.png"
        target.parent.mkdir(exist_ok=True)
        Image.new("RGBA", (4, 4), (90, 160, 210, 255)).save(target)
        targets.append(target)

    def checkpoint(index, seed, blob=None):
        path = tmp_path / "weights" / targets[index].stem / f"seed_{seed:03d}" / "checkpoints" / "best.pt"
        write_blob(path, small_blob() if blob is None else blob)
        (path.parents[1] / "best_summary.json").write_text('{"score": 0.01}')
        return path

    checkpoint(0, 1)
    checkpoint(1, 2)
    checkpoint(3, 2)
    return targets, checkpoint


@pytest.mark.parametrize("metadata", BAD_METADATA)
def test_malformed_optional_health_metadata_cannot_crash_unrelated_picker(catalog, metadata):
    targets, checkpoint = catalog
    path = checkpoint(2, 1)
    (path.parents[1] / "resolved_config.json").write_text(json.dumps(metadata))
    # A bad threshold must retain the existing .2 cutoff, not approve a failed run.
    (path.parents[1] / "best_summary.json").write_text('{"score": 0.3}')
    status = clash.target_status(targets[2], preferred_seed=1)
    assert status["status"] == "collapsed"
    assert status["selectable"] is False
    assert status["seed"] == 1
    json.dumps(status, allow_nan=False)
    world, ui = new_world(targets)
    before = freeze_world(world)
    ui.draw(dt=0)
    ui.refresh_picker_status(force=True)
    ui.action(("target", 2))
    ui.draw(dt=0)
    assert_world_unchanged(world, before)
    assert ui.status[0]["status"] == "ready"
    assert ui.status[2]["status"] == "collapsed"


def write_bad_checkpoint(path, kind):
    raw = {"empty": b"", "invalid-pickle": b"not a checkpoint", "incomplete-opcode": b"\x80"}
    if kind in raw:
        path.write_bytes(raw[kind])
        return
    blob = small_blob(channels=4 if kind == "four-channels" else 6)
    if kind == "model-null":
        blob["config"]["model"] = None
    elif kind == "data-list":
        blob["config"]["data"] = []
    elif kind == "train-null":
        blob["config"]["train"] = None
    elif kind == "missing-channels":
        del blob["config"]["model"]["channels"]
    elif kind == "wrong-shape":
        blob["model"]["fc0.weight"] = torch.zeros(1)
    elif kind == "score-null":
        blob["score"] = None
    elif kind == "nonfinite-fire-rate":
        blob["config"]["model"]["fire_rate"] = float("nan")
    write_blob(path, blob)
    if kind == "truncated-archive":
        path.write_bytes(path.read_bytes()[:80])


CORRUPT_KINDS = ["empty", "invalid-pickle", "incomplete-opcode", "truncated-archive", "model-null",
                 "data-list", "train-null", "missing-channels", "four-channels", "wrong-shape",
                 "nonfinite-fire-rate", "score-null"]


def new_world(targets, *extra):
    args = arena.parse_args(["--device", "cpu", "--cpu-threads", "1", "--grid-size", "16",
                             "--seed", "27", "--left-seed", "1", "--right-seed", "2",
                             "--round-ticks", "9", "--warmup-ticks", "2", *extra])
    world = arena.Arena(args, targets)
    ui = arena.ArenaUI(world)
    ui.paused = True
    ui.draw(dt=0)
    return world, ui


def freeze_world(world):
    return {
        "bundles": (world.left, world.right),
        "indices": (world.left_index, world.right_index),
        "pins": (world.args.left_seed, world.args.right_seed),
        "steps": world.steps, "size": world.size,
        "policy": (world.mode, world.duel_enabled, world.args.grid_size, world._deferred_right),
        "starts": copy.deepcopy(world.start_positions),
        "cache": dict(world.cache),
        "tensors": [(value, value.clone()) for value in (world.a, world.b, world.owner, world.control)],
        "duel": world.duel, "lesson": world.lesson,
        "duel_state": world.duel.snapshot() if world.duel else None,
        "lesson_state": world.lesson.snapshot() if world.lesson else None,
        "rng": rng_state(),
    }


def assert_world_unchanged(world, before):
    assert world.left is before["bundles"][0] and world.right is before["bundles"][1]
    assert (world.left_index, world.right_index) == before["indices"]
    assert (world.args.left_seed, world.args.right_seed) == before["pins"]
    assert (world.steps, world.size, world.start_positions) == (before["steps"], before["size"], before["starts"])
    assert (world.mode, world.duel_enabled, world.args.grid_size, world._deferred_right) == before["policy"]
    assert world.cache.keys() == before["cache"].keys()
    assert all(world.cache[key] is value for key, value in before["cache"].items())
    for actual, (original, expected) in zip((world.a, world.b, world.owner, world.control), before["tensors"]):
        assert actual is original
        assert torch.equal(actual, expected)
    assert world.duel is before["duel"] and world.lesson is before["lesson"]
    if world.duel:
        assert world.duel.snapshot() == before["duel_state"]
    if world.lesson:
        assert world.lesson.snapshot() == before["lesson_state"]
    assert_rng_equal(before["rng"])


@pytest.mark.parametrize("kind", CORRUPT_KINDS)
@pytest.mark.parametrize("mode,side", [("lab", 0), ("lab", 1), ("duel", 0), ("duel", 1), ("lesson", 0)])
def test_rejected_ui_selection_preserves_world_rng_and_pins_then_recovers(catalog, kind, mode, side):
    targets, checkpoint = catalog
    path = checkpoint(2, 1 if side == 0 else 2)
    write_bad_checkpoint(path, kind)
    world, ui = new_world(targets, *([] if mode == "lab" else [f"--{mode}"]))
    if world.lesson:
        world.plant_lesson()
    world.step(3)
    ui.refresh()
    ui.action("right" if side else "left")
    ui.draw(dt=0)
    before = freeze_world(world)
    # First click and then retry by number key; both real event routes must
    # recover, rather than only a mocked model-loader rejection succeeding.
    rect = next(rect for rect, action in ui.buttons if action == ("target", 2))
    ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1))
    assert_world_unchanged(world, before)
    ui.event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3,
                               mod=pygame.KMOD_SHIFT if side else 0))
    ui.draw(dt=0)
    assert_world_unchanged(world, before)
    assert ui.paused
    assert ui.message and len(ui.message) <= 1024
    assert "replayed" not in ui.message.lower()
    # Failed loads are not cached: repairing just the private file makes the
    # next selection usable without recreating the UI or changing either pin.
    write_blob(path, small_blob())
    ui.action(("target", 2))
    ui.draw(dt=0)
    selected = world.left if side == 0 else world.right
    assert selected["source"] == str(path)
    assert (world.left_index if side == 0 else world.right_index) == 2
    assert (world.args.left_seed, world.args.right_seed) == (1, 2)
    assert world.steps == 0
    if world.lesson:
        world.plant_lesson()
    assert world.step(1) == 1


@pytest.mark.parametrize("kind", ["empty", "incomplete-opcode", "model-null", "train-null", "four-channels"])
@pytest.mark.parametrize("exit_action", ["lesson", "duel"])
@pytest.mark.parametrize("pin", [None, 2])
def test_deferred_lesson_opponent_falls_back_and_retains_selection_policy(catalog, kind, exit_action, pin):
    targets, checkpoint = catalog
    path = checkpoint(2, 2)
    write_bad_checkpoint(path, kind)
    args = arena.parse_args(["--device", "cpu", "--cpu-threads", "1", "--grid-size", "16",
                             "--seed", "27", "--lesson", "--left-seed", "1", "--right", "3",
                             "--round-ticks", "9", "--warmup-ticks", "2",
                             *([] if pin is None else ["--right-seed", str(pin)])])
    world = arena.Arena(args, targets)
    ui = arena.ArenaUI(world)
    ui.draw(dt=0)
    world.plant_lesson()
    world.step(3)
    assert world._deferred_right
    ui.action(exit_action)
    ui.draw(dt=0)
    assert world.lesson is None and not world._deferred_right
    assert world.right is world.left and world.right_index == world.left_index
    assert (world.args.left_seed, world.args.right_seed) == (1, pin)
    assert "unavailable" in ui.message.lower()
    assert ("auto selection kept" if pin is None else "pin 002 kept") in ui.message.lower()
    assert (world.size, world.steps) == (16, 0)
    assert bool(world.duel) is (exit_action == "duel")
    assert world.step(1) == 1
    ui.refresh()
    ui.draw(dt=0)
    # The fallback has seed 1, but the original right-side policy must still
    # select this unrelated seed-2 checkpoint on the next explicit request.
    ui.action("right")
    ui.action(("target", 3))
    ui.draw(dt=0)
    assert world.right["name"] == targets[3].stem
    assert world.right["seed_dir"] == "seed_002"
    assert (world.args.left_seed, world.args.right_seed) == (1, pin)
    assert world.step(1) == 1


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("mode", ["lab", "duel"])
def test_unrepresentable_automatic_grid_selection_is_transactional(catalog, side, mode):
    targets, checkpoint = catalog
    blob = small_blob()
    # This dimension cannot be converted to a signed 64-bit tensor size;
    # even the old implementation rejects it before allocating any huge state.
    blob["config"]["data"]["grid_size"] = 2**63
    path = checkpoint(2, 1 if side == 0 else 2, blob)
    world, ui = new_world(targets, "--grid-size", "0", *([] if mode == "lab" else ["--duel"]))
    world.step(3)
    ui.refresh()
    ui.action("right" if side else "left")
    before = freeze_world(world)
    for _ in range(2):
        ui.action(("target", 2))
        ui.draw(dt=0)
        assert_world_unchanged(world, before)
        assert_clear_error(ui.message, "grid", "dimension", "size")
    write_blob(path, small_blob())
    ui.action(("target", 2))
    ui.draw(dt=0)
    assert world.size == 16 and world.steps == 0
    assert (world.left if side == 0 else world.right)["source"] == str(path)
    assert (world.args.left_seed, world.args.right_seed) == (1, 2)
    assert world.step(1) == 1


def test_duplicate_long_parameter_name_has_bounded_error_before_allocation(tmp_path, monkeypatch):
    blob = small_blob()
    name = "unexpected" * 800
    blob["model"][name] = torch.zeros(1)
    blob["model"]["_orig_mod." + name] = torch.zeros(1)
    path = write_blob(tmp_path / "duplicate-name.pt", blob)
    constructor = Mock(side_effect=AssertionError("invalid model state allocated a model"))
    monkeypatch.setattr(clash, "NCA", constructor)
    with pytest.raises(ValueError) as error:
        clash.load_v2_model(path, "cpu")
    assert_clear_error(error.value, "model state")
    assert isinstance(error.value.__cause__, ValueError)
    constructor.assert_not_called()
