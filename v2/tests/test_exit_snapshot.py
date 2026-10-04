"""Explicit UI snapshots reflect final queued actions, without advancing time."""

import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import pygame
import pytest
import torch

from arena import Arena, ArenaUI, parse_args
from clash import list_targets
from nca import NCA


def tiny_bundle(target, device, *args, **kwargs):
    model = NCA(channels=8, hidden_size=8, fire_rate=1).to(device).eval()
    with torch.no_grad():
        model.fc1.bias[0] = .2  # A step visibly changes the seeded cell.
    return {"model": model, "channels": 8, "grid_size": 16,
            "channels_last": False, "name": target.stem, "health": "ready",
            "seed_dir": "seed_000", "score": .01, "source": "synthetic-ui-test"}


@pytest.fixture
def world_ui(tmp_path):
    rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    args = parse_args(["--device", "cpu", "--grid-size", "16", "--seed", "17",
                       "--window-size", "860", "--snapshot", str(tmp_path / "final.png")])
    with patch("arena.ensure_model", side_effect=tiny_bundle):
        world = Arena(args, list_targets())
    ui = ArenaUI(world)
    ui.paused = True
    try:
        yield world, ui
    finally:
        pygame.quit()
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
        torch.set_rng_state(rng[0])
        np.random.set_state(rng[1])
        random.setstate(rng[2])


def key(value):
    return pygame.event.Event(pygame.KEYDOWN, key=value, mod=0)


def run_events(world, ui, events):
    draws = []
    original = ui.draw

    def capture(dt=None):
        before = (world.steps, torch.get_rng_state().clone(),
                  [t.clone() for t in (world.a, world.b, world.owner, world.control)])
        original(dt=dt)
        assert world.steps == before[0]
        assert torch.equal(torch.get_rng_state(), before[1])
        assert all(torch.equal(a, b) for a, b in zip(
            (world.a, world.b, world.owner, world.control), before[2]))
        draws.append({"tick": world.steps, "stats": dict(ui.stats_cache),
                      "pixels": pygame.image.tostring(ui.window, "RGB"),
                      "size": ui.window.get_size(), "blend": ui.blend.rendered.copy(),
                      "mode": world.mode, "colors": ui.team_colors})

    with patch.object(ui, "draw", side_effect=capture), \
            patch("pygame.event.get", return_value=events), \
            patch("pygame.mouse.get_pos", return_value=(0, 0)):
        ui.run()
    return draws


@pytest.mark.parametrize("quit_event", [pygame.QUIT, pygame.KEYDOWN])
@pytest.mark.parametrize("reduced_motion", [False, True])
def test_step_then_quit_exports_current_tick_and_unsmoothed_board(world_ui, quit_event, reduced_motion):
    world, ui = world_ui
    ui.reduced_motion = reduced_motion
    end = pygame.event.Event(pygame.QUIT) if quit_event == pygame.QUIT else key(pygame.K_ESCAPE)
    draws = run_events(world, ui, [key(pygame.K_n), end])
    assert [d["tick"] for d in draws] == [0, 1]
    assert draws[-1]["stats"] == world.stats()
    np.testing.assert_array_equal(draws[-1]["blend"], world.rgb(ui.team_colors, ui.view))
    assert not np.array_equal(draws[0]["blend"], draws[-1]["blend"])
    with Image.open(world.args.snapshot) as image:
        assert image.size == draws[-1]["size"]
        assert image.convert("RGB").tobytes() == draws[-1]["pixels"]


def test_clear_then_quit_exports_empty_field_without_extra_step(world_ui):
    world, ui = world_ui
    world.step(3)
    draws = run_events(world, ui, [key(pygame.K_c), pygame.event.Event(pygame.QUIT)])
    assert [d["tick"] for d in draws] == [3, 0]
    assert draws[-1]["stats"]["left_alive"] == draws[-1]["stats"]["right_alive"] == 0
    assert not np.count_nonzero(draws[-1]["blend"])
    with Image.open(world.args.snapshot) as image:
        assert image.convert("RGB").tobytes() == draws[-1]["pixels"]


def test_resize_step_colors_then_quit_exports_final_geometry_and_hud(world_ui):
    world, ui = world_ui
    events = [pygame.event.Event(pygame.VIDEORESIZE, w=1100, h=902),
              key(pygame.K_n), key(pygame.K_t), pygame.event.Event(pygame.QUIT)]
    draws = run_events(world, ui, events)
    assert [d["tick"] for d in draws] == [0, 0, 1]
    assert draws[-1]["size"] == (1100, 902)
    assert draws[-1]["colors"] and draws[-1]["stats"] == world.stats()
    np.testing.assert_array_equal(draws[-1]["blend"], world.rgb(True, ui.view))
    with Image.open(world.args.snapshot) as image:
        assert image.size == (1100, 902)
        assert image.convert("RGB").tobytes() == draws[-1]["pixels"]


def test_no_snapshot_does_not_add_exit_draw_or_advance_time(world_ui):
    world, ui = world_ui
    destination = world.args.snapshot
    world.args.snapshot = None
    draws = run_events(world, ui, [key(pygame.K_n), pygame.event.Event(pygame.QUIT)])
    assert [d["tick"] for d in draws] == [0]
    assert world.steps == 1
    assert not destination.exists()


def test_bounded_ui_frames_snapshot_has_no_extra_simulation_tick(world_ui):
    world, ui = world_ui
    world.args.ui_frames = 1
    ui.paused = False
    draws = run_events(world, ui, [])
    assert [d["tick"] for d in draws] == [0, 1, 1]
    assert world.steps == 1
    np.testing.assert_array_equal(draws[-1]["blend"], world.rgb(ui.team_colors, ui.view))
