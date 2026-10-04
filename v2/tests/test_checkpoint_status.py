"""Selection/status policy regressions using metadata-only checkpoint fixtures."""

import builtins
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import clash  # noqa: E402


NO_SUMMARY = object()


@pytest.fixture
def candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(clash, "V2_ROOT", tmp_path)
    target = tmp_path / "targets" / "culture.png"

    def add(seed, score=NO_SUMMARY, threshold=None, checkpoint=True):
        seed_dir = tmp_path / "weights" / target.stem / f"seed_{seed:03}"
        path = seed_dir / "checkpoints" / "best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint:
            # Deliberately not a loadable model: status must only inspect metadata.
            path.write_text("metadata-only checkpoint fixture")
        if score is not NO_SUMMARY:
            (seed_dir / "best_summary.json").write_text(json.dumps({"score": score}))
        if threshold is not None:
            (seed_dir / "resolved_config.json").write_text(json.dumps({
                "stop": {"collapsed_score": threshold},
            }))
        return path

    return target, add


def assert_policy_matches_discovery(target, preferred_seed=None, allow_unhealthy=False):
    selected = clash.discover_v2_checkpoint(target, preferred_seed, allow_unhealthy)
    status = clash.target_status(target, preferred_seed, allow_unhealthy)
    diagnostic = selected or clash.discover_v2_checkpoint(target, preferred_seed, True)
    assert status["selectable"] is (selected is not None)
    assert status["checkpoint"] == (str(diagnostic) if diagnostic is not None else None)
    if diagnostic is None:
        assert status == {"status": "missing", "score": None, "seed": None,
                          "checkpoint": None, "selectable": False}
    else:
        assert status["status"] == clash.checkpoint_health(diagnostic.parents[1])
        assert status["seed"] == clash.seed_number(diagnostic.parents[1])
        if preferred_seed is not None:
            assert status["seed"] == preferred_seed
    # Public diagnostics must serialize to strict JSON, including unknown scores.
    json.dumps(status, allow_nan=False)
    return status


@pytest.mark.parametrize("preferred_seed,allow_unhealthy,seed,health,selectable", [
    (None, False, 1, "ready", True),
    (None, True, 1, "ready", True),
    (0, False, 0, "collapsed", False),
    (0, True, 0, "collapsed", True),
    (1, False, 1, "ready", True),
    (1, True, 1, "ready", True),
    (2, False, 2, "ready", True),
    (2, True, 2, "ready", True),
    (3, False, 3, "unverified", False),
    (3, True, 3, "unverified", True),
    (99, False, None, "missing", False),
    (99, True, None, "missing", False),
])
def test_exact_policy_preserves_auto_and_pinned_selection(
        candidates, preferred_seed, allow_unhealthy, seed, health, selectable):
    target, add = candidates
    add(0, .6)
    add(1, .03)
    add(2, .04)
    add(3)
    status = assert_policy_matches_discovery(target, preferred_seed, allow_unhealthy)
    assert (status["seed"], status["status"], status["selectable"]) == (seed, health, selectable)
    assert status["score"] == {0: .6, 1: .03, 2: .04, 3: None, None: None}[seed]


def test_healthy_seed_zero_is_an_explicit_pin(candidates):
    target, add = candidates
    pinned = add(0, .08)
    add(4, .01)
    assert clash.target_status(target)["seed"] == 4
    status = assert_policy_matches_discovery(target, preferred_seed=0)
    assert status == {"status": "ready", "score": .08, "seed": 0,
                      "checkpoint": str(pinned), "selectable": True}


def test_default_and_override_select_different_health_with_per_seed_thresholds(candidates):
    target, add = candidates
    collapsed = add(0, .03, threshold=.02)
    ready = add(1, .04, threshold=.05)
    ordinary = assert_policy_matches_discovery(target)
    override = assert_policy_matches_discovery(target, allow_unhealthy=True)
    assert (ordinary["checkpoint"], ordinary["status"], ordinary["score"]) == (str(ready), "ready", .04)
    assert (override["checkpoint"], override["status"], override["score"]) == (str(collapsed), "collapsed", .03)
    assert ordinary["selectable"] and override["selectable"]


@pytest.mark.parametrize("allow_unhealthy", [False, True])
def test_auto_without_healthy_candidates_reports_best_unhealthy(candidates, allow_unhealthy):
    target, add = candidates
    best = add(2, .4)
    add(0, .6)
    add(1)
    status = assert_policy_matches_discovery(target, allow_unhealthy=allow_unhealthy)
    assert status == {"status": "collapsed", "score": .4, "seed": 2,
                      "checkpoint": str(best), "selectable": allow_unhealthy}


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("preferred_seed", [None, 0])
@pytest.mark.parametrize("allow_unhealthy", [False, True])
def test_nonfinite_score_is_unverified_and_json_safe(candidates, score, preferred_seed, allow_unhealthy):
    target, add = candidates
    add(0, score)
    status = assert_policy_matches_discovery(target, preferred_seed, allow_unhealthy)
    assert status["status"] == "unverified"
    assert status["score"] is None
    assert status["seed"] == 0
    assert status["selectable"] is allow_unhealthy


@pytest.mark.parametrize("score", [False, True])
@pytest.mark.parametrize("preferred_seed", [None, 0])
@pytest.mark.parametrize("allow_unhealthy", [False, True])
def test_boolean_score_is_unverified_and_json_safe(candidates, score, preferred_seed, allow_unhealthy):
    target, add = candidates
    path = add(0, score)
    assert clash.seed_score(path.parents[1]) == float("inf")
    status = assert_policy_matches_discovery(target, preferred_seed, allow_unhealthy)
    assert status == {"status": "unverified", "score": None, "seed": 0,
                      "checkpoint": str(path), "selectable": allow_unhealthy}


@pytest.mark.parametrize("score", [0, .01])
@pytest.mark.parametrize("allow_unhealthy", [False, True])
def test_boolean_false_cannot_outrank_a_numeric_score(candidates, score, allow_unhealthy):
    target, add = candidates
    add(0, False)
    genuine = add(1, score)
    status = assert_policy_matches_discovery(target, allow_unhealthy=allow_unhealthy)
    assert status == {"status": "ready", "score": float(score), "seed": 1,
                      "checkpoint": str(genuine), "selectable": True}


@pytest.mark.parametrize("score", [-1, 0, .01, .2, 1, 2, "-1", "0", ".01", ".2", "1", "2"])
def test_finite_numeric_scores_and_strings_keep_existing_policy(candidates, score):
    target, add = candidates
    path = add(0, score)
    numeric_score = float(score)
    health = "ready" if numeric_score < .2 else "collapsed"
    assert clash.seed_score(path.parents[1]) == numeric_score
    for allow_unhealthy in (False, True):
        status = assert_policy_matches_discovery(target, allow_unhealthy=allow_unhealthy)
        assert status == {"status": health, "score": numeric_score, "seed": 0,
                          "checkpoint": str(path),
                          "selectable": health == "ready" or allow_unhealthy}


@pytest.mark.parametrize("threshold,cutoff", [
    (False, .2), (True, .2), (.2, .2), (".2", .2),
    (.05, .05), (".05", .05), (1.0, 1.0), ("1.0", 1.0),
])
@pytest.mark.parametrize("score", [.01, .05, .2, .3, 1.0])
def test_boolean_threshold_uses_default_but_numeric_thresholds_are_preserved(
        candidates, threshold, cutoff, score):
    target, add = candidates
    path = add(0, score, threshold)
    health = "ready" if score < cutoff else "collapsed"
    for allow_unhealthy in (False, True):
        status = assert_policy_matches_discovery(target, allow_unhealthy=allow_unhealthy)
        assert status == {"status": health, "score": score, "seed": 0,
                          "checkpoint": str(path),
                          "selectable": health == "ready" or allow_unhealthy}


@pytest.mark.parametrize("summary", [None, "not JSON", '{"score": "unknown"}', '{"other": 0.01}'])
def test_missing_or_invalid_summary_stays_unverified(candidates, summary):
    target, add = candidates
    path = add(0)
    if summary is not None:
        (path.parents[1] / "best_summary.json").write_text(summary)
    for allow_unhealthy in (False, True):
        status = assert_policy_matches_discovery(target, 0, allow_unhealthy)
        assert status["status"] == "unverified"
        assert status["score"] is None


@pytest.mark.parametrize("preferred_seed", [None, 0, 99])
@pytest.mark.parametrize("allow_unhealthy", [False, True])
def test_metadata_without_checkpoint_is_missing(candidates, preferred_seed, allow_unhealthy):
    target, add = candidates
    add(0, .01, checkpoint=False)
    assert_policy_matches_discovery(target, preferred_seed, allow_unhealthy)


def test_equal_score_uses_same_seed_tiebreak_as_discovery(candidates):
    target, add = candidates
    add(5, .03)
    add(2, .03)
    assert assert_policy_matches_discovery(target)["seed"] == 2


def test_status_never_loads_models_or_imports_training(candidates, monkeypatch):
    target, add = candidates
    add(0, .6)
    add(1, .03)
    add(2)

    def unexpected(*args, **kwargs):
        pytest.fail("metadata status attempted model loading or training")

    real_import = builtins.__import__

    def guard_import(name, *args, **kwargs):
        if name == "train" or name == "trainer" or name.startswith("trainer."):
            unexpected()
        return real_import(name, *args, **kwargs)

    for name in ("ensure_model", "load_v2_model", "load_checkpoint_file", "NCA"):
        monkeypatch.setattr(clash, name, unexpected)
    monkeypatch.setattr(clash.torch, "load", unexpected)
    monkeypatch.setattr(builtins, "__import__", guard_import)
    for preferred_seed in (None, 0, 1, 2, 99):
        for allow_unhealthy in (False, True):
            assert_policy_matches_discovery(target, preferred_seed, allow_unhealthy)


@pytest.mark.parametrize("preferred_seed,health,allow_unhealthy", [
    (0, "collapsed", False),
    (2, "unverified", False),
    (99, "missing", False),
    (99, "missing", True),
])
def test_ensure_model_error_reports_exact_pinned_health_and_policy(
        candidates, monkeypatch, preferred_seed, health, allow_unhealthy):
    target, add = candidates
    add(0, .6)
    add(1, .03)
    add(2)
    status = Mock(wraps=clash.target_status)
    load = Mock(side_effect=AssertionError("rejected selection must not load a model"))
    monkeypatch.setattr(clash, "target_status", status)
    monkeypatch.setattr(clash, "load_v2_model", load)
    with pytest.raises(ValueError) as error:
        clash.ensure_model(target, "cpu", preferred_seed=preferred_seed, allow_unhealthy=allow_unhealthy)
    assert str(error.value) == (
        f"No usable checkpoint for culture seed {preferred_seed} ({health}). "
        "Change or remove the checkpoint seed pin, or choose a culture "
        f"with a usable checkpoint at seed {preferred_seed}. "
        "--list-models shows automatic choices without seed pins. "
        "Train this seed explicitly, or use --allow-unhealthy to inspect "
        "existing failed/unverified weights."
    )
    status.assert_called_once_with(target, preferred_seed, allow_unhealthy)
    load.assert_not_called()


def test_ensure_model_auto_failure_retains_unpinned_error(candidates, monkeypatch):
    target, add = candidates
    add(0, .6)
    load = Mock(side_effect=AssertionError("rejected selection must not load a model"))
    monkeypatch.setattr(clash, "load_v2_model", load)
    with pytest.raises(ValueError) as error:
        clash.ensure_model(target, "cpu")
    assert str(error.value) == (
        "No usable checkpoint for culture (collapsed). "
        "Choose a ready organism with --list-models, train it explicitly, "
        "or use --allow-unhealthy to inspect failed/unverified weights."
    )
    load.assert_not_called()


@pytest.mark.parametrize("preferred_seed,score,threshold,health", [
    (None, .03, .02, "collapsed"),
    (0, .6, .2, "collapsed"),
    (0, float("nan"), .2, "unverified"),
])
def test_unhealthy_loading_policy_and_reported_health_stay_unchanged(
        candidates, monkeypatch, preferred_seed, score, threshold, health):
    target, add = candidates
    candidate = add(0, score, threshold)
    add(1, .04, .2)
    load = Mock(return_value={})
    monkeypatch.setattr(clash, "load_v2_model", load)
    status = assert_policy_matches_discovery(target, preferred_seed, True)
    bundle = clash.ensure_model(target, "cpu", preferred_seed=preferred_seed, allow_unhealthy=True)
    load.assert_called_once_with(candidate, "cpu")
    assert status["status"] == bundle["health"] == health
    assert bundle["name"] == target.stem


@pytest.mark.parametrize("target_name,pin,health,seed,selectable", [
    ("02_star", None, "ready", 1, True),
    ("02_star", 0, "collapsed", 0, False),
    ("07_umbrella", 1, "missing", None, False),
])
def test_shipped_metadata_reproduces_pinned_picker_cases(target_name, pin, health, seed, selectable):
    target = V2_ROOT / "targets" / f"{target_name}.png"
    if not (V2_ROOT / "weights" / target_name).is_dir():
        pytest.skip("shipped checkpoint metadata is not present")
    status = assert_policy_matches_discovery(target, pin)
    assert (status["status"], status["seed"], status["selectable"]) == (health, seed, selectable)
