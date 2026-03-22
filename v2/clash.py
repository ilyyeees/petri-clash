import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from nca import NCA, make_seed, pick_device
from train import train_target
from trainer.train_v2 import resolve_config


V2_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CONFIG = V2_ROOT / "trainer" / "configs" / "single_gpu_base.toml"


def list_targets():
    return sorted((V2_ROOT / "targets").glob("*.png"))


def normalize_state_dict(state_dict):
    if not state_dict:
        return state_dict

    first_key = next(iter(state_dict))
    if not first_key.startswith("_orig_mod."):
        return state_dict

    # compiled checkpoints like to smuggle this prefix in, so strip it back out
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


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
        return float(blob["score"])
    except Exception:
        return float("inf")


def maybe_channels_last(tensor, enabled, device):
    if enabled and device == "cuda":
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def discover_v2_checkpoint(target_path, preferred_seed=None):
    target_dir = V2_ROOT / "weights" / target_path.stem
    if not target_dir.is_dir():
        return None

    if preferred_seed is not None:
        seed_dir = target_dir / f"seed_{preferred_seed:03d}"
        checkpoint_path = seed_dir / "checkpoints" / "best.pt"
        if checkpoint_path.exists():
            return checkpoint_path

    candidates = []
    for seed_dir in sorted(target_dir.glob("seed_*")):
        checkpoint_path = seed_dir / "checkpoints" / "best.pt"
        if not checkpoint_path.exists():
            continue
        candidates.append((seed_score(seed_dir), seed_number(seed_dir), checkpoint_path))

    if not candidates:
        return None

    candidates.sort(key=lambda row: (row[0], row[1]))
    return candidates[0][2]


def load_v2_model(checkpoint_path, device):
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = blob.get("config")

    if config is None:
        resolved_path = checkpoint_path.parents[1] / "resolved_config.json"
        if resolved_path.exists():
            config = json.loads(resolved_path.read_text())

    if config is None:
        raise SystemExit(f"missing config for {checkpoint_path}")

    model_cfg = config["model"]
    channels_last = bool(config.get("train", {}).get("channels_last", False))
    model = NCA(
        channels=int(model_cfg["channels"]),
        hidden_size=int(model_cfg["hidden_size"]),
        fire_rate=float(model_cfg["fire_rate"]),
    ).to(device)

    if channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    model.load_state_dict(normalize_state_dict(blob["model"]))
    model.eval()

    return {
        "model": model,
        "channels": int(model_cfg["channels"]),
        "grid_size": int(config["data"]["grid_size"]),
        "channels_last": channels_last,
        "kind": "v2",
        "source": str(checkpoint_path),
        "seed_dir": checkpoint_path.parents[1].name,
        "score": float(blob.get("score", seed_score(checkpoint_path.parents[1]))),
    }


def default_model_bundle(device):
    config = resolve_config(DEFAULT_TRAIN_CONFIG)
    model_cfg = config["model"]
    channels_last = bool(config.get("train", {}).get("channels_last", False))
    model = NCA(
        channels=int(model_cfg["channels"]),
        hidden_size=int(model_cfg["hidden_size"]),
        fire_rate=float(model_cfg["fire_rate"]),
    ).to(device)
    if channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return {
        "model": model,
        "channels": int(model_cfg["channels"]),
        "grid_size": int(config["data"]["grid_size"]),
        "channels_last": channels_last,
        "kind": "scratch",
        "source": "untrained-v2",
    }


def ensure_model(target_path, device, bootstrap_steps, preferred_seed=None):
    v2_checkpoint = discover_v2_checkpoint(target_path, preferred_seed=preferred_seed)
    if v2_checkpoint is not None:
        bundle = load_v2_model(v2_checkpoint, device)
        score_text = f" score {bundle['score']:.5f}" if bundle["score"] < float("inf") else ""
        print(f"loaded {target_path.stem} from {bundle['source']} ({bundle['seed_dir']}{score_text})")
        return bundle

    weight_path = V2_ROOT / "weights" / f"{target_path.stem}.pt"
    if weight_path.exists():
        print(
            f"ignoring legacy flat weight {weight_path.name}; "
            f"v2 only uses weights/{target_path.stem}/seed_###/checkpoints/best.pt"
        )

    if bootstrap_steps > 0:
        print(f"bootstrapping {target_path.name} with the v2 trainer for {bootstrap_steps} steps")
        checkpoint_path = train_target(
            target_path,
            steps=bootstrap_steps,
            device=device,
            group_name="clash_bootstrap",
            batch_size=32 if device == "cuda" else 8,
            pool_size=1024 if device == "cuda" else 256,
            no_compile=True,
            no_amp=device != "cuda",
        )
        bundle = load_v2_model(checkpoint_path, device)
        score_text = f" score {bundle['score']:.5f}" if bundle["score"] < float("inf") else ""
        print(f"loaded {target_path.stem} from {bundle['source']} ({bundle['seed_dir']}{score_text})")
        return bundle

    print(
        f"missing v2 weights for {target_path.stem}; "
        f"run `python v2/train.py --target {target_path}` or pass --bootstrap-steps"
    )
    return default_model_bundle(device)


def clamp(value, low, high):
    return max(low, min(high, value))


def active_grid_size(requested_size, left_bundle, right_bundle):
    if requested_size > 0:
        return requested_size
    return max(int(left_bundle["grid_size"]), int(right_bundle["grid_size"]))


def random_seed_positions(size):
    span = max(2, size // 10)
    y = clamp(size // 2 + random.randint(-span, span), 2, size - 3)
    ax = clamp(size // 4 + random.randint(-span, span), 2, size - 3)
    bx = clamp((size * 3) // 4 + random.randint(-span, span), 2, size - 3)

    if abs(ax - bx) < size // 4:
        bx = clamp(ax + size // 2, 2, size - 3)

    return ax, y, bx, y


def reset_world(size, device, left_bundle, right_bundle):
    ax, ay, bx, by = random_seed_positions(size)
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
    return state_a, state_b, owner


def crater(state_a, state_b, owner, gx, gy, radius):
    h, w = state_a.shape[-2:]
    yy = torch.arange(h, device=state_a.device).view(1, 1, h, 1)
    xx = torch.arange(w, device=state_a.device).view(1, 1, 1, w)
    mask = ((xx - gx).pow(2) + (yy - gy).pow(2)) <= radius * radius

    state_a = torch.where(mask.expand_as(state_a), torch.zeros_like(state_a), state_a)
    state_b = torch.where(mask.expand_as(state_b), torch.zeros_like(state_b), state_b)
    owner = torch.where(mask, torch.zeros_like(owner), owner)
    return state_a, state_b, owner


def clash_step(state_a, state_b, owner, model_a, model_b):
    with torch.inference_mode():
        proposed_a = model_a(state_a, steps=1)
        proposed_b = model_b(state_b, steps=1)

        alpha_a = proposed_a[:, 3:4].clamp(0.0, 1.0)
        alpha_b = proposed_b[:, 3:4].clamp(0.0, 1.0)

        owned_a = owner == 1
        owned_b = owner == 2
        near_a = F.max_pool2d(owned_a.float(), 3, stride=1, padding=1) > 0
        near_b = F.max_pool2d(owned_b.float(), 3, stride=1, padding=1) > 0

        # each side keeps its own hidden soup now, otherwise they scramble each other
        claim_a = (alpha_a > 0.05) & (owned_a | near_a)
        claim_b = (alpha_b > 0.05) & (owned_b | near_b)

        next_owner = owner.clone()
        only_a = claim_a & ~claim_b
        only_b = claim_b & ~claim_a
        both = claim_a & claim_b

        next_owner = torch.where(only_a, torch.ones_like(next_owner), next_owner)
        next_owner = torch.where(only_b, torch.full_like(next_owner, 2), next_owner)

        a_wins = alpha_a >= alpha_b
        winner_owner = torch.where(a_wins, torch.ones_like(owner), torch.full_like(owner, 2))
        next_owner = torch.where(both, winner_owner, next_owner)

        next_state_a = torch.where(
            (next_owner == 1).expand_as(proposed_a),
            proposed_a,
            torch.zeros_like(proposed_a),
        )
        next_state_b = torch.where(
            (next_owner == 2).expand_as(proposed_b),
            proposed_b,
            torch.zeros_like(proposed_b),
        )

        # tiny sparks are what turn into the ugly drift, so cut them off before they matter
        support_a = F.avg_pool2d((next_owner == 1).float(), 3, stride=1, padding=1)
        support_b = F.avg_pool2d((next_owner == 2).float(), 3, stride=1, padding=1)
        stable_a = (next_owner == 1) & (support_a >= (2.0 / 9.0))
        stable_b = (next_owner == 2) & (support_b >= (2.0 / 9.0))

        next_state_a = torch.where(stable_a.expand_as(next_state_a), next_state_a, torch.zeros_like(next_state_a))
        next_state_b = torch.where(stable_b.expand_as(next_state_b), next_state_b, torch.zeros_like(next_state_b))
        next_owner = torch.where((next_owner == 1) & ~stable_a, torch.zeros_like(next_owner), next_owner)
        next_owner = torch.where((next_owner == 2) & ~stable_b, torch.zeros_like(next_owner), next_owner)

        dead_a = F.max_pool2d(next_state_a[:, 3:4], 3, stride=1, padding=1) <= 0.1
        dead_b = F.max_pool2d(next_state_b[:, 3:4], 3, stride=1, padding=1) <= 0.1

        next_state_a = torch.where(dead_a.expand_as(next_state_a), torch.zeros_like(next_state_a), next_state_a)
        next_state_b = torch.where(dead_b.expand_as(next_state_b), torch.zeros_like(next_state_b), next_state_b)
        next_owner = torch.where(dead_a & (next_owner == 1), torch.zeros_like(next_owner), next_owner)
        next_owner = torch.where(dead_b & (next_owner == 2), torch.zeros_like(next_owner), next_owner)

    return next_state_a, next_state_b, next_owner


def compose_rgba(state_a, state_b, owner):
    rgba_a = state_a[:, :4]
    rgba_b = state_b[:, :4]
    blank = torch.zeros_like(rgba_a)
    return torch.where(
        (owner == 1).expand_as(rgba_a),
        rgba_a,
        torch.where((owner == 2).expand_as(rgba_b), rgba_b, blank),
    )


def render_surface(state_a, state_b, owner):
    rgba = compose_rgba(state_a, state_b, owner)[0].detach().cpu().clamp(0.0, 1.0)
    rgb = rgba[:3] * rgba[3:4]
    image = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(image.swapaxes(0, 1))


def select_target(index, targets):
    index = index % len(targets)
    return index, targets[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=960)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--left", type=int, default=1)
    parser.add_argument("--right", type=int, default=2)
    parser.add_argument("--left-seed", type=int)
    parser.add_argument("--right-seed", type=int)
    parser.add_argument("--bootstrap-steps", type=int, default=0)
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--headless-frames", type=int, default=0)
    args = parser.parse_args()

    targets = list_targets()
    if not targets:
        raise SystemExit(f"no targets found in {V2_ROOT / 'targets'}")

    if args.headless_frames > 0:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    flags = pygame.RESIZABLE
    if args.headless_frames > 0:
        flags |= pygame.HIDDEN
    window = pygame.display.set_mode((args.window_size, args.window_size), flags)
    pygame.display.set_caption("petri clash")
    clock = pygame.time.Clock()

    left_index, left_target = select_target(args.left - 1, targets)
    right_index, right_target = select_target(args.right - 1, targets)
    left_bundle = ensure_model(left_target, args.device, args.bootstrap_steps, preferred_seed=args.left_seed)
    right_bundle = ensure_model(right_target, args.device, args.bootstrap_steps, preferred_seed=args.right_seed)
    left_model = left_bundle["model"]
    right_model = right_bundle["model"]

    print(f"left  -> {left_target.stem} [{left_bundle['kind']}]")
    print(f"right -> {right_target.stem} [{right_bundle['kind']}]")

    paused = False
    frames = 0
    grid_size = active_grid_size(args.grid_size, left_bundle, right_bundle)
    state_a, state_b, owner = reset_world(grid_size, args.device, left_bundle, right_bundle)

    digit_keys = {
        pygame.K_1: 0,
        pygame.K_2: 1,
        pygame.K_3: 2,
        pygame.K_4: 3,
        pygame.K_5: 4,
        pygame.K_6: 5,
        pygame.K_7: 6,
        pygame.K_8: 7,
        pygame.K_9: 8,
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state_a, state_b, owner = reset_world(grid_size, args.device, left_bundle, right_bundle)
                elif event.key in digit_keys and digit_keys[event.key] < len(targets):
                    target_id = digit_keys[event.key]
                    if event.mod & pygame.KMOD_SHIFT:
                        right_index, right_target = select_target(target_id, targets)
                        right_bundle = ensure_model(
                            right_target,
                            args.device,
                            args.bootstrap_steps,
                            preferred_seed=args.right_seed,
                        )
                        right_model = right_bundle["model"]
                        print(f"right -> {right_target.stem} [{right_bundle['kind']}]")
                    else:
                        left_index, left_target = select_target(target_id, targets)
                        left_bundle = ensure_model(
                            left_target,
                            args.device,
                            args.bootstrap_steps,
                            preferred_seed=args.left_seed,
                        )
                        left_model = left_bundle["model"]
                        print(f"left  -> {left_target.stem} [{left_bundle['kind']}]")

                    grid_size = active_grid_size(args.grid_size, left_bundle, right_bundle)
                    state_a, state_b, owner = reset_world(grid_size, args.device, left_bundle, right_bundle)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                width, height = window.get_size()
                gx = int(event.pos[0] * grid_size / max(width, 1))
                gy = int(event.pos[1] * grid_size / max(height, 1))
                state_a, state_b, owner = crater(
                    state_a,
                    state_b,
                    owner,
                    gx,
                    gy,
                    radius=max(2, grid_size // 12),
                )

        if not paused:
            state_a, state_b, owner = clash_step(state_a, state_b, owner, left_model, right_model)

        surface = render_surface(state_a, state_b, owner)
        scaled = pygame.transform.scale(surface, window.get_size())
        window.fill((0, 0, 0))
        window.blit(scaled, (0, 0))
        pygame.display.flip()

        frames += 1
        if args.headless_frames > 0 and frames >= args.headless_frames:
            running = False

        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
