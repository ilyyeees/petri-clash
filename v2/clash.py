import argparse
import json
import random
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import numpy as np
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import torch

from nca import NCA, make_seed, pick_device
from checkpoints import load_checkpoint_file, normalize_state_dict, checkpoint_config, load_model_state
from battle import (clash_step, crater, momentum_owner, compose_rgba,
                    DEFAULT_PRESSURE_GAIN, DEFAULT_CONTROL_DECAY, DEFAULT_CAPTURE_THRESHOLD,
                    DEFAULT_RELEASE_THRESHOLD, DEFAULT_TIE_MARGIN)


V2_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CONFIG = V2_ROOT / "trainer" / "configs" / "single_gpu_base.toml"


def list_targets():
    return sorted((V2_ROOT / "targets").glob("*.png"))


def seed_number(seed_dir):
    name = seed_dir.name
    if not name.startswith("seed_"):
        return 10**9
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return 10**9


def seed_score(seed_dir):
    summary_path = seed_dir / "best_summary.json"
    if not summary_path.exists():
        return float("inf")

    try:
        blob = json.loads(summary_path.read_text())
        score = float(blob["score"])
        return score if math.isfinite(score) else float("inf")
    except Exception:
        return float("inf")


def maybe_channels_last(tensor, enabled, device):
    if enabled and device == "cuda":
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def checkpoint_health(seed_dir):
    """Use exported evaluation metadata; never label an unknown run as healthy."""
    score = seed_score(seed_dir)
    threshold = 0.2
    try:
        config = json.loads((seed_dir / "resolved_config.json").read_text())
        if isinstance(config, Mapping):
            stop = config.get("stop", {})
            if isinstance(stop, Mapping):
                threshold = float(stop.get("collapsed_score", threshold))
    except (OSError, ValueError, TypeError, OverflowError):
        pass
    if not math.isfinite(threshold) or threshold <= 0:
        threshold = 0.2
    if not math.isfinite(score):
        return "unverified"
    return "ready" if score < threshold else "collapsed"


def discover_v2_checkpoint(target_path, preferred_seed=None, allow_unhealthy=False):
    target_dir = V2_ROOT / "weights" / Path(target_path).stem
    candidates = []
    for seed_dir in sorted(target_dir.glob("seed_*")):
        if preferred_seed is not None and seed_number(seed_dir) != preferred_seed:
            continue
        checkpoint_path = seed_dir / "checkpoints" / "best.pt"
        if not checkpoint_path.exists():
            continue
        if checkpoint_health(seed_dir) != "ready" and not allow_unhealthy:
            continue
        candidates.append((seed_score(seed_dir), seed_number(seed_dir), checkpoint_path))
    candidates.sort(key=lambda row: (row[0], row[1]))
    return candidates[0][2] if candidates else None


def target_status(target_path, preferred_seed=None, allow_unhealthy=False):
    """Describe the exact selection policy without loading or training a model.

    Health remains independent of eligibility: an unhealthy override can make a
    collapsed or unverified checkpoint selectable, but never makes it ready.
    Rejected selections retain the same seed pin when looking up diagnostics.
    """
    checkpoint = discover_v2_checkpoint(target_path, preferred_seed, allow_unhealthy)
    selectable = checkpoint is not None
    if checkpoint is None:
        checkpoint = discover_v2_checkpoint(target_path, preferred_seed, allow_unhealthy=True)
    if checkpoint is None:
        return {"status": "missing", "score": None, "seed": None,
                "checkpoint": None, "selectable": False}
    seed_dir = checkpoint.parents[1]
    score = seed_score(seed_dir)
    return {"status": checkpoint_health(seed_dir),
            "score": score if math.isfinite(score) else None,
            "seed": seed_number(seed_dir), "checkpoint": str(checkpoint),
            "selectable": selectable}


def _play_config_number(section, key, *, integer=False, minimum=None, maximum=None):
    """Validate saved scalar settings before constructing a playable model."""
    value = section.get(key)
    try:
        if isinstance(value, bool) or not isinstance(value, (Real, str)):
            raise ValueError("expected a numeric scalar")
        number = int(value) if integer else float(value)
        if integer and not isinstance(value, str) and number != value:
            raise ValueError("expected an integer")
        if not integer and not math.isfinite(number):
            raise ValueError("expected a finite number")
        if minimum is not None and number < minimum:
            raise ValueError(f"must be at least {minimum}")
        if maximum is not None and number > maximum:
            raise ValueError(f"must be at most {maximum}")
    except (TypeError, ValueError, OverflowError) as exc:
        expected = "an integer" if integer else "a finite number"
        if minimum is not None:
            expected += f" at least {minimum}"
        if maximum is not None:
            expected += f" and at most {maximum}"
        # Conversion errors may include the entire malformed saved value.
        # Keep picker notices bounded; retain that detail in the chained cause.
        raise ValueError(f"invalid play checkpoint config {key}: expected {expected}") from exc
    return number


def _play_model_state(state_dict, channels, hidden_size):
    """Check architecture metadata against weights before allocating an NCA."""
    try:
        state_dict = normalize_state_dict(state_dict)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid play checkpoint model state") from exc
    shapes = {
        "perception_filters": (channels * 3, 1, 3, 3),
        "fc0.weight": (hidden_size, channels * 3, 1, 1),
        "fc0.bias": (hidden_size,),
        "fc1.weight": (channels, hidden_size, 1, 1),
        "fc1.bias": (channels,),
    }
    if state_dict.keys() != shapes.keys():
        raise ValueError("play checkpoint parameters do not match the NCA architecture")
    for name, shape in shapes.items():
        value = state_dict[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"play checkpoint parameter {name} does not match model config")
    return state_dict


def load_v2_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    blob = load_checkpoint_file(checkpoint_path, device)
    config = checkpoint_config(blob, checkpoint_path=checkpoint_path)
    for section in ("model", "data", "train"):
        if not isinstance(config.get(section, {} if section == "train" else None), Mapping):
            raise ValueError(f"missing or invalid {section} config for {checkpoint_path}")

    model_cfg = config["model"]
    channels = _play_config_number(model_cfg, "channels", integer=True, minimum=5)
    hidden_size = _play_config_number(model_cfg, "hidden_size", integer=True, minimum=1)
    fire_rate = _play_config_number(model_cfg, "fire_rate", minimum=0, maximum=1)
    # Match the arena's minimum supported grid, including its seed margins.
    tensor_limit = torch.iinfo(torch.int64).max
    grid_size = _play_config_number(config["data"], "grid_size", integer=True,
                                    minimum=8, maximum=tensor_limit)
    state_dict = _play_model_state(blob["model"], channels, hidden_size)
    # These are representation limits, not a promise that available memory can
    # hold a large valid arena. Account for int64 ownership and the largest NCA
    # feature map; let actual allocation/device failures propagate as before.
    float_bytes = torch.finfo(torch.get_default_dtype()).bits // 8
    bytes_per_cell = max(8, float_bytes * max(channels * 3, hidden_size))
    if grid_size * grid_size > tensor_limit // bytes_per_cell:
        raise ValueError("invalid play checkpoint config grid_size: tensor storage exceeds the signed 64-bit limit")
    channels_last = bool(config.get("train", {}).get("channels_last", False))
    model = NCA(
        channels=channels,
        hidden_size=hidden_size,
        fire_rate=fire_rate,
    ).to(device)

    if channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    load_model_state(model, state_dict)
    model.eval()
    try:
        score = float(blob.get("score", seed_score(checkpoint_path.parents[1])))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid checkpoint score: {checkpoint_path}") from exc

    return {
        "model": model,
        "channels": channels,
        "grid_size": grid_size,
        "channels_last": channels_last,
        "kind": "v2",
        "source": str(checkpoint_path),
        "seed_dir": checkpoint_path.parents[1].name,
        "score": score if math.isfinite(score) else None,
    }


def ensure_model(target_path, device, bootstrap_steps=0, preferred_seed=None, allow_unhealthy=False):
    checkpoint = discover_v2_checkpoint(target_path, preferred_seed, allow_unhealthy)
    if checkpoint is None and bootstrap_steps > 0:
        # Training is opt-in and imported only when explicitly requested.
        from train import train_target
        checkpoint = train_target(
            target_path, steps=bootstrap_steps, seed=preferred_seed or 0, device=device,
            group_name="clash_bootstrap", batch_size=32 if device == "cuda" else 8,
            pool_size=1024 if device == "cuda" else 256, no_compile=True, no_amp=device != "cuda",
        )
    if checkpoint is None:
        status = target_status(target_path, preferred_seed, allow_unhealthy)["status"]
        seed_text = f" seed {preferred_seed}" if preferred_seed is not None else ""
        if preferred_seed is None:
            guidance = (
                "Choose a ready organism with --list-models, train it explicitly, "
                "or use --allow-unhealthy to inspect failed/unverified weights."
            )
        else:
            guidance = (
                "Change or remove the checkpoint seed pin, or choose a culture "
                f"with a usable checkpoint at seed {preferred_seed}. "
                "--list-models shows automatic choices without seed pins. "
                "Train this seed explicitly, or use --allow-unhealthy to inspect "
                "existing failed/unverified weights."
            )
        raise ValueError(
            f"No usable checkpoint for {Path(target_path).stem}{seed_text} ({status}). "
            f"{guidance}"
        )
    bundle = load_v2_model(checkpoint, device)
    bundle["health"] = checkpoint_health(checkpoint.parents[1])
    bundle["name"] = Path(target_path).stem
    return bundle


def clamp(value, low, high):
    return max(low, min(high, value))


def active_grid_size(requested_size, left_bundle, right_bundle):
    if requested_size > 0:
        return requested_size
    return max(int(left_bundle["grid_size"]), int(right_bundle["grid_size"]))


def parse_grid_pos(text):
    try:
        x_text, y_text = text.split(",", 1)
        return int(x_text.strip()), int(y_text.strip())
    except Exception as exc:
        raise argparse.ArgumentTypeError("expected x,y") from exc


def random_seed_positions(size):
    span = max(2, size // 10)
    y = clamp(size // 2 + random.randint(-span, span), 2, size - 3)
    ax = clamp(size // 4 + random.randint(-span, span), 2, size - 3)
    bx = clamp((size * 3) // 4 + random.randint(-span, span), 2, size - 3)

    if abs(ax - bx) < size // 4:
        bx = clamp(ax + size // 2, 2, size - 3)

    return ax, y, bx, y


def clamp_seed_pos(pos, size):
    if pos is None:
        return None
    x, y = pos
    return clamp(int(x), 2, size - 3), clamp(int(y), 2, size - 3)


def reset_world(size, device, left_bundle, right_bundle, left_pos=None, right_pos=None):
    ax, ay, bx, by = random_seed_positions(size)
    if left_pos is not None:
        ax, ay = clamp_seed_pos(left_pos, size)
    if right_pos is not None:
        bx, by = clamp_seed_pos(right_pos, size)
    state_a = make_seed(
        1,
        channels=left_bundle["channels"],
        height=size,
        width=size,
        xs=[ax],
        ys=[ay],
        device=device,
    )
    state_b = make_seed(
        1,
        channels=right_bundle["channels"],
        height=size,
        width=size,
        xs=[bx],
        ys=[by],
        device=device,
    )
    state_a = maybe_channels_last(state_a, left_bundle["channels_last"], device)
    state_b = maybe_channels_last(state_b, right_bundle["channels_last"], device)

    owner = torch.zeros(1, 1, size, size, dtype=torch.long, device=device)
    owner[0, 0, ay, ax] = 1
    owner[0, 0, by, bx] = 2
    control = torch.zeros(1, 1, size, size, dtype=torch.float32, device=device)
    control[0, 0, ay, ax] = 1.0
    control[0, 0, by, bx] = -1.0
    return state_a, state_b, owner, control


def render_surface(state_a, state_b, owner, team_colors=False):
    import pygame
    rgba = compose_rgba(state_a, state_b, owner)[0].detach().cpu().clamp(0.0, 1.0)
    if team_colors:
        alpha = rgba[3:4]
        owner_mask = owner[0, 0].detach().cpu()
        rgb = torch.zeros_like(rgba[:3])
        rgb[0] = torch.where(owner_mask == 1, alpha[0], torch.zeros_like(alpha[0]))
        rgb[2] = torch.where(owner_mask == 2, alpha[0], torch.zeros_like(alpha[0]))
    else:
        rgb = rgba[:3] * rgba[3:4]
    image = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(image.swapaxes(0, 1))


def select_target(index, targets):
    index = index % len(targets)
    return index, targets[index]


def main():
    from arena import main as run_arena
    run_arena(default_mode="hard")


if __name__ == "__main__":
    main()
