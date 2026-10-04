"""Checkpoint regression coverage. Every test runs without a GPU."""

from collections import OrderedDict
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pytest
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from checkpoints import (  # noqa: E402
    checkpoint_config,
    load_checkpoint_file,
    load_model_state,
    normalize_state_dict,
    save_checkpoint_file,
)
from nca import NCA, make_seed  # noqa: E402
from trainer.common import checkpoint_rng_blob, seed_everything  # noqa: E402
from trainer import eval_v2  # noqa: E402
from trainer.train_v2 import (  # noqa: E402
    resolve_config,
    save_best_checkpoint,
    save_latest_checkpoint,
    try_resume,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config():
    config = resolve_config(V2_ROOT / "trainer/configs/single_gpu_base.toml")
    config["model"] = {"channels": 6, "hidden_size": 8, "fire_rate": 1.0}
    config["data"] = {"grid_size": 8, "target_size": 4}
    config["train"].update(compile=False, channels_last=False, amp=False)
    config["eval"].update(steps=[1, 2], preview_steps=[1, 2], damage_after=1,
                          recover_steps=1, recovery_preview_steps=[0, 1], preview_scale=1)
    config["runtime"]["device"] = "cpu"
    return config


def make_model():
    return NCA(channels=6, hidden_size=8, fire_rate=1.0)


def wrapped_state(model):
    return OrderedDict(
        ("_orig_mod._orig_mod." + key if i % 2 else key, value.clone())
        for i, (key, value) in enumerate(model.state_dict().items())
    )


def test_normalize_mixed_repeated_and_nested_wrappers():
    source = OrderedDict([
        ("perception_filters", torch.tensor(1)),
        ("_orig_mod._orig_mod.fc0._orig_mod.weight", torch.tensor(2)),
        ("_orig_mod.fc0.bias", torch.tensor(3)),
    ])
    source._metadata = OrderedDict([
        ("", {"version": 0}),
        ("_orig_mod", {"version": 1}),
        ("_orig_mod._orig_mod", {"version": 2}),
        ("_orig_mod._orig_mod.fc0", {"version": 3}),
    ])
    result = normalize_state_dict(source)
    assert list(result) == ["perception_filters", "fc0.weight", "fc0.bias"]
    assert result._metadata == {"": {"version": 2}, "fc0": {"version": 3}}
    assert list(source)[1].startswith("_orig_mod.")


def test_normalize_rejects_colliding_names():
    with pytest.raises(ValueError, match="duplicate checkpoint parameter"):
        normalize_state_dict({"fc0.bias": torch.tensor(1), "_orig_mod.fc0.bias": torch.tensor(2)})


@pytest.mark.parametrize("compile_source", [False, True])
@pytest.mark.parametrize("compile_target", [False, True])
def test_plain_and_compiled_models_interoperate(compile_source, compile_target):
    source = make_model()
    target = make_model()
    with torch.no_grad():
        source.fc1.weight.fill_(0.01)
    if compile_source:
        source = torch.compile(source, backend="eager")
    if compile_target:
        target = torch.compile(target, backend="eager")
    load_model_state(target, source.state_dict())
    seed = make_seed(1, channels=6, height=8)
    with torch.inference_mode():
        torch.testing.assert_close(target(seed), source(seed))


def test_normalized_weights_load_into_compiled_child():
    source = make_model()
    target = make_model()
    target.fc0 = torch.compile(target.fc0, backend="eager")
    load_model_state(target, wrapped_state(source))
    for name, value in normalize_state_dict(target.state_dict()).items():
        torch.testing.assert_close(value, source.state_dict()[name])


def test_config_merges_companion_and_saved_values_without_mutation(tmp_path):
    path = tmp_path / "checkpoints" / "best.pt"
    defaults = {"model": {"channels": 16, "hidden_size": 128}, "runtime": {"device": "cpu"}}
    (tmp_path / "resolved_config.json").write_text(json.dumps({
        "model": {"channels": 20}, "data": {"grid_size": 40},
    }))
    saved = {"config": {"model": {"channels": 24}, "data": {"grid_size": 48}}}
    result = checkpoint_config(saved, defaults, checkpoint_path=path)
    assert result["model"] == {"channels": 24, "hidden_size": 128}
    assert result["data"]["grid_size"] == 48
    result["model"]["channels"] = 99
    assert saved["config"]["model"]["channels"] == 24
    assert defaults["model"]["channels"] == 16
    assert checkpoint_config({}, checkpoint_path=path)["model"]["channels"] == 20


def test_eval_loader_honors_saved_architecture_and_fire_rate(tmp_path):
    path = tmp_path / "best.pt"
    saved_config = small_config()
    saved_config["model"]["fire_rate"] = 0.75
    torch.save({"model": wrapped_state(make_model()), "config": saved_config}, path)
    defaults = small_config()
    defaults["model"].update(channels=16, hidden_size=128, fire_rate=0.1)
    model, _ = eval_v2.load_checkpoint(path, defaults, "cpu")
    assert (model.channels, model.hidden_size, model.fire_rate) == (6, 8, 0.75)
    assert not model.training


@pytest.mark.parametrize("separate_checkpoint", [False, True])
def test_eval_cli_uses_saved_data_eval_and_target(tmp_path, monkeypatch, capsys, separate_checkpoint):
    from PIL import Image

    path = tmp_path / "checkpoints" / "best.pt"
    path.parent.mkdir()
    target = tmp_path / "target.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(target)
    config = small_config()
    config["target"] = str(target)
    # Simulate a GPU-trained checkpoint evaluated on a CPU-only machine.
    config["runtime"]["device"] = "cuda"
    if separate_checkpoint:
        (tmp_path / "resolved_config.json").write_text(json.dumps(config))
        path = tmp_path / "external" / "best.pt"
        path.parent.mkdir()
        torch.save({"model": wrapped_state(make_model()), "step": 17}, path)
    else:
        torch.save({"model": wrapped_state(make_model()), "config": config, "step": 17}, path)
    defaults = small_config()
    defaults["data"] = {"grid_size": 24, "target_size": 20}
    defaults["eval"]["steps"] = [77]
    monkeypatch.setattr(eval_v2, "resolve_config", lambda _: defaults)
    out = tmp_path / "preview.png"
    argv = ["eval_v2", "--run-dir", str(tmp_path), "--device", "cpu", "--out", str(out)]
    if separate_checkpoint:
        argv.extend(["--checkpoint", str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    eval_v2.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["checkpoint_step"] == 17
    assert [point["step"] for point in summary["points"]] == [1, 2]
    assert Path(summary["target"]) == target
    assert out.is_file() and out.with_suffix(".json").is_file()
    assert Image.open(out).width == 4 * 8 + 3 * 4


def training_parts(compiled=False):
    model = make_model()
    if compiled:
        model = torch.compile(model, backend="eager")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    sum(param.square().sum() for param in model.parameters()).backward()
    optimizer.step()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.step()
    scheduler.step()
    return model, optimizer, scaler, scheduler


@pytest.mark.parametrize("compile_source", [False, True])
@pytest.mark.parametrize("compile_target", [False, True])
def test_save_resume_preserves_weights_optimizer_scheduler_pool_and_rng(tmp_path, compile_source, compile_target):
    source, optimizer, scaler, scheduler = training_parts(compile_source)
    pool = torch.rand(3, 6, 8, 8)
    seed_everything(1234)
    save_latest_checkpoint(tmp_path, source, optimizer, scaler, scheduler, pool, 19, 0.25, small_config())
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    path = tmp_path / "checkpoints" / "latest.pt"
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert all("_orig_mod" not in key for key in blob["model"])
    assert isinstance(blob["rng"]["numpy"][1], list)
    target, target_optimizer, target_scaler, target_scheduler = training_parts(compile_target)
    result = try_resume(tmp_path, target, target_optimizer, target_scaler, target_scheduler, "cpu")
    assert result["step"] == 19 and result["best_score"] == 0.25
    torch.testing.assert_close(result["pool"], pool, rtol=0, atol=0)
    assert target_scheduler.state_dict() == scheduler.state_dict()
    assert target_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for src_state, dst_state in zip(optimizer.state.values(), target_optimizer.state.values()):
        for key in src_state:
            torch.testing.assert_close(dst_state[key], src_state[key])
    for name, value in normalize_state_dict(target.state_dict()).items():
        torch.testing.assert_close(value, normalize_state_dict(source.state_dict())[name])
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    torch.testing.assert_close(torch.rand(3), expected_rng[2], rtol=0, atol=0)


def test_legacy_numpy_rng_checkpoint_load_and_resume(tmp_path):
    model, optimizer, scaler, scheduler = training_parts()
    seed_everything(54)
    blob = {
        "model": wrapped_state(model), "optimizer": optimizer.state_dict(),
        "scaler": None, "scheduler": scheduler.state_dict(),
        "pool": torch.rand(2, 6, 8, 8).half(), "step": 7, "best_score": 0.125,
        "rng": checkpoint_rng_blob(), "config": small_config(),
    }
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    path = tmp_path / "checkpoints" / "latest.pt"
    path.parent.mkdir()
    torch.save(blob, path)
    with pytest.raises(pickle.UnpicklingError):
        torch.load(path, weights_only=True)
    loaded = load_checkpoint_file(path)
    assert isinstance(loaded["rng"]["numpy"][1], np.ndarray)
    target, target_optimizer, target_scaler, target_scheduler = training_parts(True)
    result = try_resume(tmp_path, target, target_optimizer, target_scaler, target_scheduler, "cpu")
    assert result["step"] == 7
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    torch.testing.assert_close(torch.rand(3), expected_rng[2], rtol=0, atol=0)


def test_best_save_has_plain_weights(tmp_path):
    model = torch.compile(make_model(), backend="eager")
    save_best_checkpoint(tmp_path, model, 10, 0.1, {"score": 0.1}, small_config())
    blob = load_checkpoint_file(tmp_path / "checkpoints" / "best.pt")
    assert all("_orig_mod" not in key for key in blob["model"])
    assert blob["step"] == 10
    assert json.loads((tmp_path / "best_summary.json").read_text())["score"] == 0.1


def test_atomic_save_failure_preserves_existing_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    save_checkpoint_file({"model": make_model().state_dict(), "step": 1}, path)
    original = path.read_bytes()

    def failing_save(blob, handle):
        handle.write(b"incomplete checkpoint")
        raise OSError("disk write interrupted")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint_file({"model": {}, "step": 2}, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


class UnsupportedCheckpointValue:
    pass


def test_loader_does_not_fall_back_to_unrestricted_pickle(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save({"model": {}, "unsupported": UnsupportedCheckpointValue()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint_file(path)


def test_all_distributed_checkpoints_are_loadable():
    paths = sorted((V2_ROOT / "weights").glob("*/seed_*/checkpoints/best.pt"))
    if not paths:
        pytest.skip("no distributed checkpoints in this checkout")
    for path in paths:
        model, blob = eval_v2.load_checkpoint(path, small_config(), "cpu")
        assert model.channels == blob["config"]["model"]["channels"]
        assert model.hidden_size == blob["config"]["model"]["hidden_size"]
