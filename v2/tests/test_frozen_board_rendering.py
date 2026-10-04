"""Paused raster reuse preserves the original renderer's pixels and semantics.

All models are synthetic, CPU-only, and bounded. These tests exercise the real
SDL dummy display, frame blending, controllers, overlays, event dispatch, and
exit export; they do not claim trained-model quality or native display speed.
"""

import gc
import os
from pathlib import Path
import random
import sys
import weakref
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
import torch.nn.functional as F

import arena
import clash
from lesson import LessonConfig


class GrowingPatch(torch.nn.Module):
    """A small repairable patch with stochastic hidden channels."""

    def __init__(self, color):
        super().__init__()
        self.color, self.calls, self.repair, self.invalid = color, 0, True, False

    def forward(self, state, steps=1):
        assert steps == 1
        self.calls += 1
        life = state[:, 3:4] > .1
        if self.repair:
            life = F.max_pool2d(life.float(), 3, stride=1, padding=1) > 0
        h, w = state.shape[-2:]
        window = torch.zeros_like(life)
        window[:, :, h // 2 - 5:h // 2 + 6, w // 2 - 5:w // 2 + 6] = True
        life = (life & window).float()
        result = torch.zeros_like(state)
        result[:, :3] = life * self.color
        result[:, 3:5] = life
        result[:, 5:] = (state[:, 5:] * .8 + torch.rand_like(state[:, 5:]) * .02) * life
        if self.invalid:
            result[0, -1, h // 2, w // 2] = float("nan")
        return result


def synthetic_bundle(target, device, *args, **kwargs):
    color = .3 + sum(map(ord, target.stem)) % 40 / 100
    return {"model": GrowingPatch(color).to(device).eval(), "channels": 8,
            "grid_size": 16, "channels_last": False, "name": target.stem,
            "health": "ready", "seed_dir": "seed_000", "score": .01,
            "source": "synthetic-frozen-raster-test"}


@pytest.fixture
def make_ui():
    rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()

    def create(*extra):
        args = arena.parse_args(["--device", "cpu", "--cpu-threads", "1",
                                 "--grid-size", "16", "--seed", "27",
                                 "--window-size", "860", *extra])
        world = arena.Arena(args, clash.list_targets())
        return world, arena.ArenaUI(world)

    with patch("arena.ensure_model", side_effect=synthetic_bundle), \
            patch("pygame.mouse.get_pos", return_value=(-1, -1)), \
            patch("pygame.mouse.get_pressed", return_value=(False, False, False)):
        try:
            yield create
        finally:
            pygame.quit()
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic)
            torch.set_rng_state(rng[0])
            np.random.set_state(rng[1])
            random.setstate(rng[2])


def pixels(surface):
    return pygame.image.tostring(surface, "RGB")


def assert_pixels(actual, expected):
    np.testing.assert_array_equal(np.frombuffer(actual, dtype=np.uint8),
                                  np.frombuffer(expected, dtype=np.uint8))


def uncached_surface(ui, rgb):
    # This is the complete pre-cache board raster path, with no cache access.
    field = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    return pygame.transform.scale(field, ui.board.size)


def hitboxes(ui):
    return (tuple(ui.board), tuple(ui.score_rect),
            tuple(tuple(rect) for rect in ui.score_bars),
            tuple((tuple(rect), action) for rect, action in ui.buttons),
            tuple((tuple(rect), action) for rect, action in ui.result_buttons),
            None if ui.result_rect is None else tuple(ui.result_rect))


def simulation_state(world):
    return {"tensors": [value.clone() for value in (world.a, world.b, world.owner, world.control)],
            "steps": world.steps, "torch": torch.get_rng_state().clone(),
            "numpy": np.random.get_state(), "python": random.getstate(),
            "calls": [bundle["model"].calls for bundle in world.cache.values()],
            "duel": world.duel.snapshot() if world.duel else None,
            "lesson": world.lesson.snapshot() if world.lesson else None}


def assert_simulation_unchanged(world, before):
    after = simulation_state(world)
    for actual, expected in zip(after.pop("tensors"), before["tensors"]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(after.pop("torch"), before["torch"])
    numpy = after.pop("numpy")
    assert numpy[0] == before["numpy"][0]
    np.testing.assert_array_equal(numpy[1], before["numpy"][1])
    assert numpy[2:] == before["numpy"][2:]
    for key, actual in after.items():
        assert actual == before[key]


def assert_draw_matches_reference(world, ui, dt=0):
    """Check the whole window, controls, inputs, and overlay source ownership."""
    before = simulation_state(world)
    rasters, targets = [], []
    original_raster, original_rgb = ui.board_surface, world.rgb

    def target(*args, **kwargs):
        rgb = original_rgb(*args, **kwargs)
        targets.append((rgb, rgb.copy()))
        return rgb

    def cached(rgb):
        original = rgb.copy()
        surface = original_raster(rgb)
        rasters.append((rgb, original, surface, pixels(surface)))
        return surface

    with patch.object(world, "rgb", side_effect=target), \
            patch.object(ui, "board_surface", side_effect=cached):
        ui.draw(dt=dt)
    actual, controls = pixels(ui.window), hitboxes(ui)
    for rgb, original, surface, untouched in rasters:
        np.testing.assert_array_equal(rgb, original)
        assert_pixels(pixels(surface), untouched)
        assert_pixels(untouched, pixels(uncached_surface(ui, original)))
    # A zero-time redraw leaves the post-blend RGB and animation age unchanged.
    # Only the two original raster operations replace the implementation.
    with patch.object(ui, "board_surface", side_effect=lambda rgb: uncached_surface(ui, rgb)):
        ui.draw(dt=0)
    assert_pixels(actual, pixels(ui.window))
    assert hitboxes(ui) == controls
    for rgb, original in targets:
        np.testing.assert_array_equal(rgb, original)
    assert_simulation_unchanged(world, before)
    return actual


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (16, 16)])
@pytest.mark.parametrize("size", [(1, 1), (31, 23), (518, 518)])
@pytest.mark.parametrize("layout", ["contiguous", "reversed", "readonly"])
def test_exact_raster_orientation_scaling_and_input_ownership(make_ui, shape, size, layout):
    _, ui = make_ui()
    ui.paused = True
    ui.board.size = size
    rgb = np.arange(shape[0] * shape[1] * 3, dtype=np.uint8).reshape(*shape, 3)
    if layout == "reversed":
        rgb = rgb[::-1, ::-1]
    elif layout == "readonly":
        rgb.flags.writeable = False
    original = rgb.copy()
    expected = pixels(uncached_surface(ui, rgb))
    with patch("pygame.surfarray.make_surface", wraps=pygame.surfarray.make_surface) as make, \
            patch("pygame.transform.scale", wraps=pygame.transform.scale) as scale:
        first = ui.board_surface(rgb)
        for _ in range(3):
            assert ui.board_surface(rgb.copy()) is first
        assert make.call_count == scale.call_count == 1
    assert_pixels(pixels(first), expected)
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_array_equal(ui._scaled_board_rgb, original)
    assert not np.shares_memory(ui._scaled_board_rgb, rgb)
    assert ui._scaled_board_rgb.dtype == np.uint8
    assert ui._scaled_board_rgb.nbytes == rgb.nbytes


def test_live_frames_never_compare_or_copy_and_release_paused_cache(make_ui):
    class NoCopyArray(np.ndarray):
        def copy(self, *args, **kwargs):
            raise AssertionError("Live board rendering must not copy an RGB cache")

    _, ui = make_ui()
    ui.board.size = (40, 35)
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    ui.paused = True
    first = ui.board_surface(rgb)
    assert ui.board_surface(rgb) is first
    ui.paused = False
    with patch("arena.np.array_equal", side_effect=AssertionError("Live cache comparison")), \
            patch("pygame.surfarray.make_surface", wraps=pygame.surfarray.make_surface) as make, \
            patch("pygame.transform.scale", wraps=pygame.transform.scale) as scale:
        for _ in range(3):
            assert ui.board_surface(rgb.view(NoCopyArray)) is not first
            assert ui._scaled_board is ui._scaled_board_rgb is ui._scaled_board_size is None
        assert make.call_count == scale.call_count == 3
    ui.paused = True
    resumed_pause = ui.board_surface(rgb)
    assert resumed_pause is not first
    assert ui.board_surface(rgb) is resumed_pause


def test_one_entry_cache_detects_in_place_single_byte_changes_and_releases_old_entries(make_ui):
    _, ui = make_ui()
    ui.paused = True
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    ui.board.size = (40, 40)
    old = ui.board_surface(rgb)
    saved = ui._scaled_board_rgb
    old_pixels = pixels(old)
    rgb[3, 4, 2] = 1
    assert_pixels(pixels(old), old_pixels)
    assert saved[3, 4, 2] == 0  # The key cannot alias caller-owned memory.
    new = ui.board_surface(rgb)
    assert new is not old
    assert_pixels(pixels(new), pixels(uncached_surface(ui, rgb)))
    stale_rgb, stale_surface = weakref.ref(saved), weakref.ref(old)
    del old, saved
    gc.collect()
    assert stale_rgb() is stale_surface() is None
    for value in range(2, 20):
        old_rgb = weakref.ref(ui._scaled_board_rgb)
        rgb[3, 4, 2] = value
        ui.board_surface(rgb)
        assert old_rgb() is None
    previous = ui._scaled_board
    ui.board.size = (41, 40)  # Width-only change must invalidate an equal image.
    assert ui.board_surface(rgb) is not previous
    previous = ui._scaled_board
    ui.board.size = (41, 42)  # So must a height-only change.
    assert ui.board_surface(rgb) is not previous
    assert ui._scaled_board.get_size() == (41, 42)
    previous = ui._scaled_board
    reshaped = rgb.reshape(4, 16, 3)
    assert ui.board_surface(reshaped) is not previous
    assert ui._scaled_board_rgb.shape == reshaped.shape
    assert_pixels(pixels(ui._scaled_board), pixels(uncached_surface(ui, reshaped)))


def test_repeated_paused_frames_and_pause_resume_preserve_full_window(make_ui):
    world, ui = make_ui()
    world.step(3)
    for paused in (False, True, True, True, False, False, True, True):
        ui.paused = paused
        assert_draw_matches_reference(world, ui, .03)
        if not paused:
            assert ui._scaled_board is ui._scaled_board_rgb is ui._scaled_board_size is None


def test_paused_easing_keys_on_post_blend_rgb_until_pixels_settle(make_ui):
    world, ui = make_ui()
    ui.paused = True
    black = np.zeros((world.size, world.size, 3), dtype=np.uint8)
    white = np.full_like(black, 240)
    with patch.object(world, "rgb", return_value=black):
        ui.draw(dt=0)
    initial = ui._scaled_board
    with patch.object(world, "rgb", return_value=white):
        previous = ui._scaled_board_rgb.copy()
        for dt in (.01, .02, .04, .08, .1):
            assert_draw_matches_reference(world, ui, dt)
            expected = np.rint(ui.blend.rendered).clip(0, 255).astype(np.uint8)
            np.testing.assert_array_equal(ui._scaled_board_rgb, expected)
            assert not np.array_equal(previous, expected)
            assert ui._scaled_board is not initial
            assert not np.array_equal(expected, white)
            previous, initial = expected.copy(), ui._scaled_board
        for _ in range(10):
            assert_draw_matches_reference(world, ui, .1)
        np.testing.assert_array_equal(ui._scaled_board_rgb, white)
        settled = ui._scaled_board
        assert_draw_matches_reference(world, ui, .1)
        assert ui._scaled_board is settled
    assert not black.any()
    assert np.all(white == 240)


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("reduced_motion", [False, True])
def test_step_damage_plant_clear_and_both_culture_selections(make_ui, mode, reduced_motion):
    world, ui = make_ui("--mode", mode)
    ui.paused, ui.reduced_motion = True, reduced_motion
    assert_draw_matches_reference(world, ui)
    actions = [lambda: ui.action("step"), lambda: ui.interact(8, 8),
               lambda: ui.interact(7, 7, 0), lambda: ui.interact(9, 9, 1),
               lambda: ui.action("clear"), lambda: ui.interact(8, 8, 0),
               lambda: ui.action("step"), lambda: ui.action(("target", 2)),
               lambda: ui.action("right"), lambda: ui.action(("target", 3)),
               lambda: ui.action("reset")]
    for action in actions:
        action()
        assert ui.paused
        assert_draw_matches_reference(world, ui, .04)
        assert_draw_matches_reference(world, ui, .04)
    assert world.left_index == 2 and world.right_index == 3


@pytest.mark.parametrize("size", [(860, 740), (1100, 902)])
@pytest.mark.parametrize("view", ["organisms", "territory", "pressure"])
@pytest.mark.parametrize("colors", [False, True])
@pytest.mark.parametrize("reduced_motion", [False, True])
def test_resize_view_colors_fx_and_pointer_overlays(make_ui, size, view, colors, reduced_motion):
    world, ui = make_ui()
    world.step(5)
    ui.paused, ui.team_colors, ui.reduced_motion = True, colors, reduced_motion
    ui.draw(dt=0)  # Populate a cache before resizing or changing the view.
    ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=size[0], h=size[1]))
    ui.action(("view", view))
    ui.interact(8, 8)
    ui.interact(7, 7, 0)
    ui.interact(9, 9, 1)
    for tool in ("damage", "plant_left", "plant_right"):
        ui.action(("tool", tool))
        with patch("pygame.mouse.get_pos", return_value=ui.board.center):
            assert_draw_matches_reference(world, ui, .1)
    assert ui.effects
    surface = ui._scaled_board
    for _ in range(6):
        assert_draw_matches_reference(world, ui, .25)
    assert not ui.effects
    assert ui._scaled_board is surface



def test_view_color_motion_and_rule_toggles_never_reuse_stale_pixels(make_ui):
    world, ui = make_ui()
    world.step(3)
    ui.paused = True
    assert_draw_matches_reference(world, ui)
    for action in (("view", "territory"), "colors", ("view", "pressure"),
                   "motion", ("view", "organisms"), "colors", "motion",
                   "mode", "step", "colors", "motion", "mode", "step"):
        ui.action(action)
        assert ui.paused
        assert_draw_matches_reference(world, ui, .04)
        assert_draw_matches_reference(world, ui, .04)


def reach_lesson(world, phase):
    model = world.left["model"]
    world.start_lesson(LessonConfig(grow_ticks=7, recovery_ticks=16, hold_ticks=3))
    if phase != "seed":
        world.plant_lesson()
    if phase == "grow":
        world.step(1)
    elif phase in ("unavailable", "invalid"):
        model.repair = phase != "unavailable"
        model.invalid = phase == "invalid"
        world.step(10_000)
    elif phase not in ("seed", "grow"):
        world.step(10_000)
        if phase != "damage":
            world.cut_lesson()
        if phase not in ("damage", "injured"):
            world.watch_lesson()
            model.repair = phase != "timeout"
            world.step(1 if phase == "recover" else 10_000)
    assert world.lesson.phase == phase


@pytest.mark.parametrize("phase", ["seed", "grow", "damage", "injured", "recover",
                                   "complete", "timeout", "unavailable", "invalid"])
@pytest.mark.parametrize("reduced_motion", [False, True])
def test_all_lesson_stages_and_cue_overlays(make_ui, phase, reduced_motion):
    world, ui = make_ui()
    ui.paused = True
    assert_draw_matches_reference(world, ui)
    reach_lesson(world, phase)
    ui.reduced_motion = reduced_motion
    ui.refresh(reset=True)
    for dt in (0, .05, .25):
        assert_draw_matches_reference(world, ui, dt)
    assert ui.paused
    assert ui._scaled_board_rgb.shape == (48, 48, 3)


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("reduced_motion", [False, True])
def test_finished_duel_visible_hidden_result_and_rematch(make_ui, mode, reduced_motion):
    world, ui = make_ui("--mode", mode, "--duel", "--round-ticks", "9", "--warmup-ticks", "2")
    ui.reduced_motion = reduced_motion
    assert_draw_matches_reference(world, ui)
    world.step(100)
    assert world.duel.finished
    assert not ui.paused
    assert_draw_matches_reference(world, ui, .1)
    assert ui.paused and ui.result_rect and ui.result_buttons
    for _ in range(8):
        assert_draw_matches_reference(world, ui, .1)
    surface = ui._scaled_board
    for visible in (False, True, False):
        ui.action("result")
        assert ui.result_visible is visible
        assert_draw_matches_reference(world, ui)
        assert ui._scaled_board is surface
        assert bool(ui.result_rect) is visible
        assert bool(ui.result_buttons) is visible
    ui.action("reset")
    assert not ui.paused and world.steps == 0 and not world.duel.finished
    assert_draw_matches_reference(world, ui)
    assert ui._scaled_board is ui._scaled_board_rgb is ui._scaled_board_size is None


@pytest.mark.parametrize("reduced_motion", [False, True])
@pytest.mark.parametrize("last_action", [pygame.K_n, pygame.K_c, pygame.K_t])
def test_final_queued_action_exit_snapshot_is_uncached_exact(make_ui, tmp_path, reduced_motion, last_action):
    destination = tmp_path / "final.png"
    world, ui = make_ui("--snapshot", str(destination))
    world.step(3)
    ui.paused, ui.reduced_motion = True, reduced_motion
    ui.draw(dt=0)
    snapshots = []
    original = ui.draw

    def checked_draw(dt=None):
        before = simulation_state(world)
        original(dt=dt)
        actual = pixels(ui.window)
        with patch.object(ui, "board_surface", side_effect=lambda rgb: uncached_surface(ui, rgb)):
            original(dt=0)
        assert_pixels(actual, pixels(ui.window))
        assert_simulation_unchanged(world, before)
        snapshots.append((actual, ui.window.get_size(), ui._scaled_board_rgb.copy()))

    events = [pygame.event.Event(pygame.VIDEORESIZE, w=1100, h=902),
              pygame.event.Event(pygame.KEYDOWN, key=last_action, mod=0),
              pygame.event.Event(pygame.QUIT)]
    with patch.object(ui, "draw", side_effect=checked_draw), \
            patch("pygame.event.get", return_value=events):
        ui.run()
    assert len(snapshots) == 3  # Initial, resize, and final zero-time export.
    assert world.steps == (4 if last_action == pygame.K_n else 0 if last_action == pygame.K_c else 3)
    np.testing.assert_array_equal(snapshots[-1][2], world.rgb(ui.team_colors, ui.view))
    with Image.open(destination) as image:
        assert image.size == snapshots[-1][1] == (1100, 902)
        assert_pixels(image.convert("RGB").tobytes(), snapshots[-1][0])
