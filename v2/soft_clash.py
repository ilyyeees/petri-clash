import argparse
import os

import numpy as np
import pygame
import torch

from clash import (
    V2_ROOT,
    active_grid_size,
    clamp,
    ensure_model,
    list_targets,
    parse_grid_pos,
    select_target,
)
from nca import make_seed, pick_device


def maybe_channels_last(tensor, enabled, device):
    if enabled and device == "cuda":
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def blank_state(bundle, size, device):
    state = torch.zeros(1, bundle["channels"], size, size, device=device)
    return maybe_channels_last(state, bundle["channels_last"], device)


def seed_state(bundle, size, device, pos=None):
    if pos is None:
        state = make_seed(1, channels=bundle["channels"], height=size, width=size, device=device)
    else:
        x, y = clamp_seed_pos(pos, size)
        state = make_seed(
            1,
            channels=bundle["channels"],
            height=size,
            width=size,
            xs=[x],
            ys=[y],
            device=device,
        )
    return maybe_channels_last(state, bundle["channels_last"], device)


def add_seed(state, x, y):
    if state.shape[1] <= 4:
        return state
    state[0, 3, y, x] = 1.0
    state[0, 4, y, x] = 1.0
    return state


def reset_world(size, device, left_bundle, right_bundle, left_pos=None, right_pos=None):
    state_a = seed_state(left_bundle, size, device, left_pos)
    state_b = seed_state(right_bundle, size, device, right_pos)
    return state_a, state_b


def clamp_seed_pos(pos, size):
    x, y = pos
    return clamp(int(x), 0, size - 1), clamp(int(y), 0, size - 1)


def crater(state_a, state_b, gx, gy, radius):
    h, w = state_a.shape[-2:]
    yy = torch.arange(h, device=state_a.device).view(1, 1, h, 1)
    xx = torch.arange(w, device=state_a.device).view(1, 1, 1, w)
    mask = ((xx - gx).pow(2) + (yy - gy).pow(2)) <= radius * radius
    state_a = torch.where(mask.expand_as(state_a), torch.zeros_like(state_a), state_a)
    state_b = torch.where(mask.expand_as(state_b), torch.zeros_like(state_b), state_b)
    return state_a, state_b


def growth_step(state_a, state_b, model_a, model_b):
    with torch.inference_mode():
        next_a = model_a(state_a, steps=1)
        next_b = model_b(state_b, steps=1)
    return next_a, next_b


def compose_rgba(state_a, state_b):
    rgba_a = state_a[:, :4].clamp(0.0, 1.0)
    rgba_b = state_b[:, :4].clamp(0.0, 1.0)
    alpha_a = rgba_a[:, 3:4]
    alpha_b = rgba_b[:, 3:4]

    premul_a = rgba_a[:, :3] * alpha_a
    premul_b = rgba_b[:, :3] * alpha_b
    out_alpha = (alpha_a + alpha_b * (1.0 - alpha_a)).clamp(0.0, 1.0)
    out_rgb = premul_a + premul_b * (1.0 - alpha_a)
    out_rgb = torch.where(out_alpha > 1e-6, out_rgb / out_alpha.clamp_min(1e-6), out_rgb)

    return torch.cat([out_rgb, out_alpha], dim=1).clamp(0.0, 1.0)


def render_surface(state_a, state_b, team_colors=False):
    if team_colors:
        alpha_a = state_a[:, 3:4].detach().cpu().clamp(0.0, 1.0)[0, 0]
        alpha_b = state_b[:, 3:4].detach().cpu().clamp(0.0, 1.0)[0, 0]
        rgb = torch.zeros(3, alpha_a.shape[0], alpha_a.shape[1])
        rgb[0] = alpha_a
        rgb[2] = alpha_b
    else:
        rgba = compose_rgba(state_a, state_b)[0].detach().cpu()
        rgb = rgba[:3] * rgba[3:4]

    image = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(image.swapaxes(0, 1))


def alive_cells(state):
    return int((state[:, 3:4] > 0.1).sum().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=960)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--left", type=int, default=1)
    parser.add_argument("--right", type=int, default=2)
    parser.add_argument("--left-seed", type=int)
    parser.add_argument("--right-seed", type=int)
    parser.add_argument("--left-pos", type=parse_grid_pos)
    parser.add_argument("--right-pos", type=parse_grid_pos)
    parser.add_argument("--team-colors", action="store_true")
    parser.add_argument("--crater-radius", type=int, default=4)
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
    pygame.display.set_caption("petri soft clash")
    clock = pygame.time.Clock()

    left_index, left_target = select_target(args.left - 1, targets)
    right_index, right_target = select_target(args.right - 1, targets)
    left_bundle = ensure_model(left_target, args.device, args.bootstrap_steps, preferred_seed=args.left_seed)
    right_bundle = ensure_model(right_target, args.device, args.bootstrap_steps, preferred_seed=args.right_seed)
    left_model = left_bundle["model"]
    right_model = right_bundle["model"]

    print(f"left  -> {left_target.stem} [{left_bundle['kind']}]")
    print(f"right -> {right_target.stem} [{right_bundle['kind']}]")
    print("controls: space pause, r reset, c clear, t colors, [ ] crater size")
    print("controls: left click crater, shift+left seed left, right click seed right")

    paused = False
    frames = 0
    grid_size = active_grid_size(args.grid_size, left_bundle, right_bundle)
    crater_radius = max(1, int(args.crater_radius))
    state_a, state_b = reset_world(
        grid_size,
        args.device,
        left_bundle,
        right_bundle,
        left_pos=args.left_pos,
        right_pos=args.right_pos,
    )

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
                    state_a, state_b = reset_world(
                        grid_size,
                        args.device,
                        left_bundle,
                        right_bundle,
                        left_pos=args.left_pos,
                        right_pos=args.right_pos,
                    )
                elif event.key == pygame.K_c:
                    state_a = blank_state(left_bundle, grid_size, args.device)
                    state_b = blank_state(right_bundle, grid_size, args.device)
                elif event.key == pygame.K_t:
                    args.team_colors = not args.team_colors
                elif event.key == pygame.K_LEFTBRACKET:
                    crater_radius = max(1, crater_radius - 1)
                    print(f"crater radius -> {crater_radius}")
                elif event.key == pygame.K_RIGHTBRACKET:
                    crater_radius += 1
                    print(f"crater radius -> {crater_radius}")
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
                    state_a, state_b = reset_world(
                        grid_size,
                        args.device,
                        left_bundle,
                        right_bundle,
                        left_pos=args.left_pos,
                        right_pos=args.right_pos,
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                width, height = window.get_size()
                gx = int(event.pos[0] * grid_size / max(width, 1))
                gy = int(event.pos[1] * grid_size / max(height, 1))
                gx, gy = clamp(gx, 0, grid_size - 1), clamp(gy, 0, grid_size - 1)
                mods = pygame.key.get_mods()

                if event.button == 1 and mods & pygame.KMOD_SHIFT:
                    add_seed(state_a, gx, gy)
                elif event.button == 3 and mods & pygame.KMOD_SHIFT:
                    add_seed(state_b, gx, gy)
                elif event.button == 1:
                    state_a, state_b = crater(state_a, state_b, gx, gy, crater_radius)
                elif event.button == 3:
                    add_seed(state_b, gx, gy)

        if not paused:
            state_a, state_b = growth_step(state_a, state_b, left_model, right_model)

        surface = render_surface(state_a, state_b, team_colors=args.team_colors)
        scaled = pygame.transform.scale(surface, window.get_size())
        window.blit(scaled, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

        frames += 1
        if args.headless_frames > 0 and frames >= args.headless_frames:
            running = False

    print(f"alive left={alive_cells(state_a)} right={alive_cells(state_b)}")
    pygame.quit()


if __name__ == "__main__":
    main()
