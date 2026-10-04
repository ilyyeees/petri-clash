import argparse
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

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
    from arena import main as run_arena
    run_arena(default_mode="soft")


if __name__ == "__main__":
    main()
