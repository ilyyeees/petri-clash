import argparse
import os
import random
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from nca import NCA, make_seed, pick_device
from train import train_target


def list_targets():
    return sorted(Path("targets").glob("*.png"))


def load_model_blob(blob, device):
    model = NCA(
        channels=blob.get("channels", 16),
        hidden_size=blob.get("hidden_size", 128),
        fire_rate=blob.get("fire_rate", 0.5),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def load_model(weight_path, device):
    blob = torch.load(weight_path, map_location=device)
    return load_model_blob(blob, device)


def ensure_model(target_path, device, bootstrap_steps):
    weight_path = Path("weights") / f"{target_path.stem}.pt"
    if weight_path.exists():
        blob = torch.load(weight_path, map_location="cpu")
        trained_steps = int(blob.get("train_steps", 0) or 0)
        if trained_steps >= bootstrap_steps or bootstrap_steps <= 0:
            print(f"loaded {weight_path.name} ({trained_steps} steps)")
            return load_model_blob(blob, device)
        print(
            f"refreshing {weight_path.name} because it only has "
            f"{trained_steps} steps"
        )

    if bootstrap_steps > 0:
        print(f"bootstrapping {target_path.name} for {bootstrap_steps} steps")
        train_target(
            target_path,
            out_path=weight_path,
            steps=bootstrap_steps,
            pool_size=256,
            batch_size=4,
            min_rollout=32,
            max_rollout=48,
            save_every=bootstrap_steps,
            device=device,
        )
        return load_model(weight_path, device)

    print(f"missing {weight_path.name}, using an untrained rule")
    model = NCA().to(device)
    model.eval()
    return model


def clamp(value, low, high):
    return max(low, min(high, value))


def random_seed_positions(size):
    span = max(2, size // 10)
    y = clamp(size // 2 + random.randint(-span, span), 2, size - 3)
    ax = clamp(size // 4 + random.randint(-span, span), 2, size - 3)
    bx = clamp((size * 3) // 4 + random.randint(-span, span), 2, size - 3)

    if abs(ax - bx) < size // 4:
        bx = clamp(ax + size // 2, 2, size - 3)

    return ax, y, bx, y


def reset_world(size, device):
    ax, ay, bx, by = random_seed_positions(size)
    state = torch.zeros(1, 16, size, size, device=device)
    owner = torch.zeros(1, 1, size, size, dtype=torch.long, device=device)

    seed_a = make_seed(1, height=size, width=size, xs=[ax], ys=[ay], device=device)
    seed_b = make_seed(1, height=size, width=size, xs=[bx], ys=[by], device=device)
    state = torch.maximum(seed_a, seed_b)

    owner[0, 0, ay, ax] = 1
    owner[0, 0, by, bx] = 2
    return state, owner


def crater(state, owner, gx, gy, radius):
    h, w = state.shape[-2:]
    yy = torch.arange(h, device=state.device).view(1, 1, h, 1)
    xx = torch.arange(w, device=state.device).view(1, 1, 1, w)
    mask = ((xx - gx).pow(2) + (yy - gy).pow(2)) <= radius * radius

    state = torch.where(mask.expand_as(state), torch.zeros_like(state), state)
    owner = torch.where(mask, torch.zeros_like(owner), owner)
    return state, owner


def clash_step(state, owner, model_a, model_b):
    with torch.inference_mode():
        proposed_a = model_a(state, steps=1)
        proposed_b = model_b(state, steps=1)

        alpha_a = proposed_a[:, 3:4].clamp(0.0, 1.0)
        alpha_b = proposed_b[:, 3:4].clamp(0.0, 1.0)

        owned_a = owner == 1
        owned_b = owner == 2
        near_a = F.max_pool2d(owned_a.float(), 3, stride=1, padding=1) > 0
        near_b = F.max_pool2d(owned_b.float(), 3, stride=1, padding=1) > 0

        frontier = (near_a & near_b) | ((owner == 0) & (near_a | near_b))
        active = torch.maximum(alpha_a, alpha_b) > 0.05

        next_state = state.clone()
        next_state = torch.where(owned_a.expand_as(next_state), proposed_a, next_state)
        next_state = torch.where(owned_b.expand_as(next_state), proposed_b, next_state)

        a_wins = alpha_a >= alpha_b
        winner_state = torch.where(a_wins.expand_as(next_state), proposed_a, proposed_b)
        next_state = torch.where((frontier & active).expand_as(next_state), winner_state, next_state)

        next_owner = owner.clone()
        winner_owner = torch.where(a_wins, torch.ones_like(owner), torch.full_like(owner, 2))
        next_owner = torch.where(frontier & active, winner_owner, next_owner)

        dead = F.max_pool2d(next_state[:, 3:4], 3, stride=1, padding=1) <= 0.1
        next_state = torch.where(dead.expand_as(next_state), torch.zeros_like(next_state), next_state)
        next_owner = torch.where(dead, torch.zeros_like(next_owner), next_owner)

    return next_state, next_owner


def render_surface(state):
    rgba = state[0, :4].detach().cpu().clamp(0.0, 1.0)
    rgb = rgba[:3] * rgba[3:4]
    image = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(image.swapaxes(0, 1))


def select_target(index, targets):
    index = index % len(targets)
    return index, targets[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=960)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--left", type=int, default=1)
    parser.add_argument("--right", type=int, default=2)
    parser.add_argument("--bootstrap-steps", type=int, default=500)
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--headless-frames", type=int, default=0)
    args = parser.parse_args()

    targets = list_targets()
    if not targets:
        raise SystemExit("no targets found in targets/")

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
    left_model = ensure_model(left_target, args.device, args.bootstrap_steps)
    right_model = ensure_model(right_target, args.device, args.bootstrap_steps)

    print(f"left  -> {left_target.stem}")
    print(f"right -> {right_target.stem}")

    paused = False
    frames = 0
    state, owner = reset_world(args.grid_size, args.device)

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
                    state, owner = reset_world(args.grid_size, args.device)
                elif event.key in digit_keys and digit_keys[event.key] < len(targets):
                    target_id = digit_keys[event.key]
                    if event.mod & pygame.KMOD_SHIFT:
                        right_index, right_target = select_target(target_id, targets)
                        right_model = ensure_model(right_target, args.device, args.bootstrap_steps)
                        print(f"right -> {right_target.stem}")
                    else:
                        left_index, left_target = select_target(target_id, targets)
                        left_model = ensure_model(left_target, args.device, args.bootstrap_steps)
                        print(f"left  -> {left_target.stem}")
                    state, owner = reset_world(args.grid_size, args.device)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                width, height = window.get_size()
                gx = int(event.pos[0] * args.grid_size / max(width, 1))
                gy = int(event.pos[1] * args.grid_size / max(height, 1))
                state, owner = crater(state, owner, gx, gy, radius=max(2, args.grid_size // 12))

        if not paused:
            state, owner = clash_step(state, owner, left_model, right_model)

        surface = render_surface(state)
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
