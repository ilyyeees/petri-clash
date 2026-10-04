"""CPU/dummy-SDL contracts for persistent, mouse-only sandbox tools.

Tiny synthetic NCAs and ready checkpoint metadata isolate input routing,
layout, edit feedback, and lifecycle guards. These tests do not evaluate model
quality, read checkpoint weights, or claim native desktop/accessibility QA.
"""

import copy
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import numpy as np
import pygame
import torch

import arena
import clash
from lesson import LessonConfig
from nca import NCA


TOOLS = ("damage", "plant_left", "plant_right")
TOOL_ACTIONS = tuple(("tool", tool) for tool in TOOLS)
LAB_ACTIONS = (*TOOL_ACTIONS, ("radius", -1), ("radius", 1), "clear")
LESSON_PHASES = ("seed", "grow", "damage", "injured", "recover", "complete",
                 "timeout", "unavailable", "invalid")
SHORT_LESSON = LessonConfig(grow_ticks=3, recovery_ticks=4, hold_ticks=2)


def ready_status(target, preferred_seed=None, allow_unhealthy=False):
    seed = 0 if preferred_seed is None else preferred_seed
    return {"status": "ready", "score": .01, "seed": seed,
            "checkpoint": f"synthetic/{target.stem}/seed_{seed:03d}/checkpoints/best.pt",
            "selectable": True}


def synthetic_bundle(target, device, *args, preferred_seed=None, **kwargs):
    status = ready_status(target, preferred_seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sum(map(ord, target.stem)) + status["seed"])
        model = NCA(channels=8, hidden_size=16).to(device).eval()
    return {"model": model, "channels": 8, "grid_size": 16,
            "channels_last": False, "name": target.stem, "health": "ready",
            "seed_dir": f"seed_{status['seed']:03d}", "score": .01,
            "kind": "v2", "source": status["checkpoint"]}


class LabToolsUIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.original_rng = self.random_states()
        self.threads = torch.get_num_threads()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.loader_patch = patch("arena.ensure_model", side_effect=synthetic_bundle)
        self.status_patch = patch("arena.target_status", side_effect=ready_status)
        self.loader = self.loader_patch.start()
        self.status_loader = self.status_patch.start()
        self.mods_patch = patch("pygame.key.get_mods", return_value=0)
        self.mods = self.mods_patch.start()
        self.new_world()

    def tearDown(self):
        pygame.quit()
        self.mods_patch.stop()
        self.status_patch.stop()
        self.loader_patch.stop()
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(self.deterministic)
        torch.set_rng_state(self.original_rng[0])
        np.random.set_state(self.original_rng[1])
        random.setstate(self.original_rng[2])

    def new_world(self, *extra):
        args = arena.parse_args([
            "--device", "cpu", "--cpu-threads", "1", "--grid-size", "16",
            "--seed", "27", "--round-ticks", "7", "--warmup-ticks", "2",
            "--left-seed", "0", "--right-seed", "1", *extra])
        self.world = arena.Arena(args, clash.list_targets())
        self.ui = arena.ArenaUI(self.world)
        self.ui.draw(dt=0)

    @staticmethod
    def random_states():
        return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()

    def assert_rng_equal(self, expected):
        actual = self.random_states()
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertEqual(actual[1][0], expected[1][0])
        np.testing.assert_array_equal(actual[1][1], expected[1][1])
        self.assertEqual(actual[1][2:], expected[1][2:])
        self.assertEqual(actual[2], expected[2])

    def snapshot(self):
        return {
            "bundles": (self.world.left, self.world.right),
            "indices": (self.world.left_index, self.world.right_index),
            "pins": (self.world.args.left_seed, self.world.args.right_seed),
            "policy": (self.world.args.allow_unhealthy, self.world.args.bootstrap_steps),
            "seed": self.world.args.seed,
            "steps": self.world.steps,
            "mode": self.world.mode,
            "tensors": [(tensor, tensor.clone()) for tensor in
                        (self.world.a, self.world.b, self.world.owner, self.world.control)],
            "parameters": [[tensor.clone() for tensor in bundle["model"].state_dict().values()]
                           for bundle in (self.world.left, self.world.right)],
            "duel": self.world.duel.snapshot() if self.world.duel else None,
            "lesson": self.world.lesson.snapshot() if self.world.lesson else None,
            "rng": self.random_states(),
        }

    def assert_frozen(self, before):
        self.assertIs(self.world.left, before["bundles"][0])
        self.assertIs(self.world.right, before["bundles"][1])
        self.assertEqual((self.world.left_index, self.world.right_index), before["indices"])
        self.assertEqual((self.world.args.left_seed, self.world.args.right_seed), before["pins"])
        self.assertEqual((self.world.args.allow_unhealthy, self.world.args.bootstrap_steps), before["policy"])
        self.assertEqual(self.world.args.seed, before["seed"])
        self.assertEqual(self.world.steps, before["steps"])
        self.assertEqual(self.world.mode, before["mode"])
        for actual, (original, value) in zip(
                (self.world.a, self.world.b, self.world.owner, self.world.control), before["tensors"]):
            self.assertIs(actual, original)
            # Byte equality also preserves the exact NaN pattern of invalid
            # soft-duel fixtures; torch.equal treats unchanged NaNs as unequal.
            self.assertTrue(torch.equal(actual.view(torch.uint8), value.view(torch.uint8)))
        for bundle, values in zip((self.world.left, self.world.right), before["parameters"]):
            for actual, value in zip(bundle["model"].state_dict().values(), values):
                self.assertTrue(torch.equal(actual, value))
        self.assertEqual(self.world.duel.snapshot() if self.world.duel else None, before["duel"])
        self.assertEqual(self.world.lesson.snapshot() if self.world.lesson else None, before["lesson"])
        self.assert_rng_equal(before["rng"])

    def key(self, key, *, up=False, mod=0, repeat=False):
        self.ui.event(pygame.event.Event(pygame.KEYUP if up else pygame.KEYDOWN,
                                       key=key, mod=mod, repeat=repeat))

    def click(self, pos, button=1):
        self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))

    def control(self, action):
        return next(rect for rect, value in self.ui.buttons if value == action)

    def click_control(self, action):
        self.click(self.control(action).center)

    def cell(self, x, y):
        return (self.ui.board.x + int((x + .5) * self.ui.board.width / self.world.size),
                self.ui.board.y + int((y + .5) * self.ui.board.height / self.world.size))

    def assert_scores(self, left, right):
        self.assertEqual(self.ui.stats_cache, self.world.stats())
        self.assertEqual((self.ui.summary["left"], self.ui.summary["right"]), (left, right))

    def reach_lesson(self, phase):
        """Use real public gates with stationary synthetic tissue, without training."""
        self.world.start_lesson(SHORT_LESSON)
        if phase == "invalid":
            self.world.lesson.invalidate("Synthetic fixture for locked invalid phase")
        elif phase != "seed":
            self.world.plant_lesson()
            if phase == "grow":
                self.world.step(1)
            elif phase == "unavailable":
                self.world.step(100)
            else:
                self.fill_lesson_patch()
                self.world.step(100)
                if phase != "damage":
                    self.world.cut_lesson()
                if phase not in ("damage", "injured"):
                    self.world.watch_lesson()
                    if phase == "complete":
                        self.fill_lesson_patch()
                    self.world.step(1 if phase == "recover" else 100)
        self.ui.refresh(reset=True)
        self.ui.draw(dt=0)
        self.assertEqual(self.world.lesson.phase, phase)

    def fill_lesson_patch(self):
        self.world.a = self.world.a.clone()
        center = self.world.size // 2
        self.world.a[:, 3:, center - 5:center + 6, center - 5:center + 6] = 1

    def test_default_damage_and_full_mouse_only_edit_sequence(self):
        for mode in ("hard", "soft"):
            for paused in (False, True):
                for reduced in (False, True):
                    with self.subTest(mode=mode, paused=paused, reduced_motion=reduced):
                        self.new_world("--mode", mode, *(["--reduced-motion"] if reduced else []))
                        self.assertEqual(self.ui.pointer_tool, "damage")
                        if paused:
                            self.click_control("pause")
                        self.world.step(2)  # Also exercise editing inference tensors.
                        self.ui.refresh()
                        self.click_control("clear")
                        self.assert_scores(0, 0)
                        self.assertEqual(self.ui.effects, [])
                        self.assertEqual(self.world.steps, 0)
                        self.assertEqual(self.ui.paused, paused)
                        while self.ui.radius > 1:
                            self.click_control(("radius", -1))
                        self.click_control(("tool", "plant_left"))
                        self.click(self.cell(4, 5))
                        self.assert_scores(1, 0)
                        self.assertEqual(self.world.a[0, 3:5, 5, 4].tolist(), [1, 1])
                        self.assertIn("LEFT", self.ui.effects[-1]["label"])
                        self.assertEqual(self.ui.effects[-1]["kind"], "plant")
                        self.click_control(("tool", "plant_right"))
                        self.click(self.cell(4, 5))
                        self.assert_scores(0 if mode == "hard" else 1, 1)
                        self.assertEqual(self.world.b[0, 3:5, 5, 4].tolist(), [1, 1])
                        self.assertIn("RIGHT", self.ui.effects[-1]["label"])
                        self.click_control(("tool", "damage"))
                        self.click(self.cell(4, 5))
                        self.assert_scores(0, 0)
                        self.assertEqual(self.ui.effects[-1]["kind"], "damage")
                        self.assertIn(f"L -{0 if mode == 'hard' else 1} / R -1 life",
                                      self.ui.effects[-1]["label"])
                        self.assertIsNone(self.ui.blend.rendered)
                        self.assertEqual(self.ui.paused, paused)
                        self.assertEqual(self.ui.side, 0)
                        self.assertEqual(self.world.steps, 0)
                        self.ui.draw(dt=0)
                        np.testing.assert_array_equal(self.ui.blend.rendered, self.world.rgb())

    def test_tool_selection_is_presentation_only_and_independent_of_checkpoint_picker(self):
        self.world.step(3)
        self.ui.refresh()
        for side in ("left", "right"):
            self.click_control(side)
            self.ui.paused = side == "right"
            before = self.snapshot()
            policy = self.ui.picker_policy()
            statuses = copy.deepcopy(self.ui.status)
            radius, paused, selected_side = self.ui.radius, self.ui.paused, self.ui.side
            with patch.object(self.world, "select", side_effect=AssertionError("tool became target")), \
                    patch.object(self.world, "load", side_effect=AssertionError("tool loaded model")):
                for tool in (*TOOLS, "plant_left", "plant_left"):
                    self.click_control(("tool", tool))
                    self.assertEqual(self.ui.pointer_tool, tool)
                    self.assertEqual((self.ui.radius, self.ui.paused, self.ui.side),
                                     (radius, paused, selected_side))
                    self.assertEqual(self.ui.picker_policy(), policy)
                    self.assertEqual(self.ui.status, statuses)
                    self.assert_frozen(before)
        self.assertEqual(self.world.left["seed_dir"], "seed_000")
        self.assertEqual(self.world.right["seed_dir"], "seed_001")

    def test_tool_persists_through_all_sandbox_controls_and_culture_changes(self):
        for tool in TOOLS:
            with self.subTest(tool=tool):
                self.ui.action(("tool", tool))
                for action in ("clear", "reset", "speed", "colors", "motion", "pause",
                               ("view", "territory"), ("view", "pressure"), ("view", "organisms"),
                               "mode", "mode", "right", ("target", 2), "left", ("target", 5)):
                    self.ui.action(action)
                    self.assertEqual(self.ui.pointer_tool, tool, action)
                self.assertEqual((self.world.args.left_seed, self.world.args.right_seed), (0, 1))
                self.assertEqual(self.world.left["seed_dir"], "seed_000")
                self.assertEqual(self.world.right["seed_dir"], "seed_001")

    def test_returning_to_a_fresh_lab_resets_tool_for_duel_and_lesson(self):
        for tool in ("plant_left", "plant_right"):
            for transition in ("duel", "lesson"):
                with self.subTest(tool=tool, transition=transition):
                    self.new_world()
                    self.click_control(("tool", tool))
                    self.click_control(transition)
                    # A transition may leave this event batch's lab rectangles.
                    self.ui.action(transition)
                    self.assertIsNone(self.world.lesson)
                    self.assertIsNone(self.world.duel)
                    self.assertEqual(self.ui.pointer_tool, "damage")
                    self.assertEqual(self.world.steps, 0)
                    self.assertFalse(self.ui.paused)
                    with patch.object(self.ui, "interact") as interact:
                        self.click(self.cell(3, 4))
                    interact.assert_called_once_with(3, 4)

    def test_mouse_radius_controls_clamp_and_keep_tool_pause_state_and_simulation(self):
        for tool in TOOLS:
            for paused in (False, True):
                with self.subTest(tool=tool, paused=paused):
                    self.ui.action(("tool", tool))
                    self.ui.paused = paused
                    before = self.snapshot()
                    for delta, expected in ((-1, 1), (1, self.world.size), (-1, 1)):
                        for _ in range(self.world.size + 2):
                            self.click_control(("radius", delta))
                        self.assertEqual(self.ui.radius, expected)
                        self.assertEqual(self.ui.pointer_tool, tool)
                        self.assertEqual(self.ui.paused, paused)
                        self.assert_frozen(before)
                    self.key(pygame.K_RIGHTBRACKET)
                    self.assertEqual(self.ui.radius, 2)
                    self.key(pygame.K_LEFTBRACKET)
                    self.assertEqual(self.ui.radius, 1)
                    self.assertEqual(self.ui.pointer_tool, tool)
                    self.assertEqual(self.ui.paused, paused)
                    self.assert_frozen(before)

    def test_clear_button_and_c_shortcut_preserve_selected_tool_and_pause(self):
        for tool in TOOLS:
            for paused in (False, True):
                for input_kind in ("mouse", "keyboard", "direct"):
                    with self.subTest(tool=tool, paused=paused, input=input_kind):
                        self.ui.action(("tool", tool))
                        self.ui.paused = paused
                        self.ui.interact(3, 4, 0)
                        if input_kind == "mouse":
                            self.click_control("clear")
                        elif input_kind == "keyboard":
                            self.key(pygame.K_c)
                        else:
                            self.ui.action("clear")
                        self.assert_scores(0, 0)
                        self.assertEqual(self.ui.pointer_tool, tool)
                        self.assertEqual(self.ui.paused, paused)
                        self.assertEqual(self.ui.effects, [])
                        self.assertEqual(self.world.steps, 0)

    def test_explicit_interact_api_keeps_damage_and_side_semantics(self):
        for tool in TOOLS:
            with self.subTest(tool=tool):
                self.ui.action(("tool", tool))
                self.ui.action("clear")
                self.ui.interact(3, 4, 0)
                self.assert_scores(1, 0)
                self.ui.interact(3, 4, 1)
                self.assert_scores(0, 1)
                self.ui.interact(3, 4)
                self.assert_scores(0, 0)
                self.assertEqual(self.ui.pointer_tool, tool)

    def test_queued_tool_click_shift_repeats_releases_and_focus_loss(self):
        with patch.object(self.ui, "interact") as interact:
            for tool in TOOLS:
                self.click_control(("tool", tool))
                for _ in range(3):
                    self.key(pygame.K_LSHIFT, repeat=True)
                self.key(pygame.K_RSHIFT)
                self.click(self.cell(3, 4))
                self.assertEqual(interact.call_args.args, (3, 4, 0))
                self.key(pygame.K_LSHIFT, up=True)
                self.click(self.cell(3, 4))
                self.assertEqual(interact.call_args.args, (3, 4, 0))
                self.key(pygame.K_RSHIFT, up=True)
                for _ in range(3):
                    self.click(self.cell(3, 4))
                    self.assertEqual(interact.call_args.args,
                                     (3, 4) if tool == "damage" else (3, 4, 0 if tool == "plant_left" else 1))
                self.key(pygame.K_LSHIFT)
                self.ui.event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
                self.assertEqual(self.ui.held_shift_keys, set())
                self.click(self.cell(3, 4))
                self.assertEqual(interact.call_args.args,
                                 (3, 4) if tool == "damage" else (3, 4, 0 if tool == "plant_left" else 1))
                self.click(self.cell(3, 4), button=3)
                self.assertEqual(interact.call_args.args, (3, 4, 1))
                self.assertEqual(self.ui.pointer_tool, tool)

    def test_event_shift_and_right_click_override_without_changing_persistent_tool(self):
        for tool in TOOLS:
            with self.subTest(tool=tool), patch.object(self.ui, "interact") as interact:
                self.ui.action(("tool", tool))
                # Physical state is deliberately contradictory: only events
                # already processed may affect this click's temporary tool.
                self.mods.return_value = pygame.KMOD_SHIFT
                self.assertEqual(self.ui.effective_pointer_tool(), tool)
                self.key(pygame.K_LSHIFT)
                self.mods.return_value = 0
                self.assertEqual(self.ui.effective_pointer_tool(), "plant_left")
                self.click(self.cell(3, 4))
                interact.assert_called_once_with(3, 4, 0)
                self.assertEqual(self.ui.effective_pointer_tool(button=3), "plant_right")
                self.click(self.cell(3, 4), button=3)
                self.assertEqual(interact.call_args.args, (3, 4, 1))
                for button in (2, 4, 5):
                    self.assertIsNone(self.ui.effective_pointer_tool(button=button))
                    self.click(self.cell(3, 4), button=button)
                self.assertEqual(interact.call_count, 2)
                self.assertEqual(self.ui.pointer_tool, tool)
                self.key(pygame.K_LSHIFT, up=True, mod=pygame.KMOD_SHIFT)
                self.mods.return_value = pygame.KMOD_SHIFT
                self.assertEqual(self.ui.effective_pointer_tool(), tool)

    def test_nonfield_clicks_never_dispatch_grid_edits(self):
        with patch.object(self.ui, "interact") as interact, \
                patch.object(self.world, "damage") as damage, \
                patch.object(self.world, "plant") as plant:
            for action in (*TOOL_ACTIONS, ("radius", -1), ("radius", 1), "left", "right",
                           ("view", "organisms"), ("view", "territory"), ("view", "pressure")):
                self.click_control(action)
            for pos in ((30, 15), (self.ui.rail_x + 5, 305),
                        (self.ui.board.right + 8, self.ui.board.centery),
                        (self.ui.board.left, self.ui.board.bottom + 5)):
                for button in (1, 3):
                    self.click(pos, button)
            interact.assert_not_called()
            damage.assert_not_called()
            plant.assert_not_called()

    def test_minimum_and_default_layouts_fit_labels_and_keep_existing_geometry(self):
        for width, height in ((860, 740), (1100, 902)):
            with self.subTest(width=width, height=height):
                self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
                with patch.object(self.ui, "button", wraps=self.ui.button) as button:
                    self.ui.draw(dt=0)
                window = self.ui.window.get_rect()
                expected_size = min(width - 286 - 56, height - 110 - 52)
                expected_x = max(20, (width - expected_size - 16 - 286) // 2)
                self.assertEqual(self.ui.board, pygame.Rect(expected_x, 110, expected_size, expected_size))
                self.assertEqual(self.ui.rail_x - self.ui.board.right, 16)
                for i, (rect, action) in enumerate(self.ui.buttons):
                    self.assertTrue(window.contains(rect), action)
                    for other, other_action in self.ui.buttons[i + 1:]:
                        self.assertFalse(rect.colliderect(other), (action, other_action))
                calls = {call.args[2]: call for call in button.call_args_list}
                for action in LAB_ACTIONS:
                    call = calls[action]
                    rect, label = pygame.Rect(call.args[0]), call.args[1]
                    self.assertLessEqual(self.ui.button_font.size(label)[0], rect.width - 6, label)
                    self.assertLessEqual(self.ui.button_font.get_height(), rect.height, label)
                    self.assertGreaterEqual(rect.left, self.ui.rail_x)
                    self.assertGreater(rect.top, self.control(("view", "organisms")).bottom)
                    self.assertFalse(rect.colliderect(self.ui.board))
                self.assertEqual(calls[("tool", "damage")].args[1], "DAMAGE")
                self.assertEqual(calls[("tool", "plant_left")].args[1], "PLANT LEFT")
                self.assertEqual(calls[("tool", "plant_right")].args[1], "PLANT RIGHT")
                self.assertIn("CLEAR", calls["clear"].args[1])
                self.assertIn("C", calls["clear"].args[1])
                for index in range(len(self.world.targets)):
                    row, col = divmod(index, 3)
                    self.assertEqual(self.control(("target", index)),
                                     pygame.Rect(self.ui.rail_x + 10 + col * 90, 366 + row * 62, 86, 57))
                for offset, view in ((0, "organisms"), (90, "territory"), (180, "pressure")):
                    self.assertEqual(self.control(("view", view)),
                                     pygame.Rect(self.ui.rail_x + 10 + offset, 580, 86, 27))

    def test_resize_then_tool_and_field_click_work_in_same_event_batch(self):
        for width, height in ((200, 200), (1100, 902), (1800, 740), (860, 1200)):
            with self.subTest(width=width, height=height):
                self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
                self.click_control(("tool", "plant_right"))
                with patch.object(self.ui, "interact") as interact:
                    self.click(self.cell(3, 4))
                interact.assert_called_once_with(3, 4, 1)
                self.click_control(("tool", "damage"))
                with patch.object(self.ui, "interact") as interact:
                    self.click(self.cell(3, 4))
                interact.assert_called_once_with(3, 4)

    def test_preview_and_dispatch_share_tool_and_snapped_cell_for_every_override(self):
        pos = (self.cell(3, 4)[0] + 2, self.cell(3, 4)[1] - 3)
        expected_cell = self.ui.pointer_cell(pos)
        scale = self.ui.board.width / self.world.size
        expected_center = (round(self.ui.board.x + (expected_cell[0] + .5) * scale),
                           round(self.ui.board.y + (expected_cell[1] + .5) * scale))
        for tool in TOOLS:
            for override in ("none", "held_left_shift", "held_right_shift", "both_shifts",
                             "physical_shift_only", "right"):
                for reduced in (False, True):
                    with self.subTest(tool=tool, override=override, reduced_motion=reduced):
                        self.ui.action(("tool", tool))
                        self.ui.reduced_motion = reduced
                        self.ui.event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
                        self.ui.event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
                        self.mods.return_value = pygame.KMOD_SHIFT if override == "physical_shift_only" else 0
                        if override in ("held_left_shift", "both_shifts", "right"):
                            self.key(pygame.K_LSHIFT)
                        if override in ("held_right_shift", "both_shifts"):
                            self.key(pygame.K_RSHIFT)
                        button = 3 if override == "right" else 1
                        expected = ("plant_right" if override == "right" else "plant_left"
                                    if override in ("held_left_shift", "held_right_shift", "both_shifts") else tool)
                        before = self.snapshot()
                        old_clip = self.ui.window.get_clip()
                        font = Mock(wraps=self.ui.fonts[12])
                        with patch("pygame.mouse.get_pos", return_value=pos), \
                                patch("pygame.mouse.get_pressed", return_value=(False, False, button == 3)), \
                                patch.object(self.ui, "effective_pointer_tool", wraps=self.ui.effective_pointer_tool) as resolve, \
                                patch("pygame.draw.circle", wraps=pygame.draw.circle) as circle, \
                                patch("pygame.draw.line", wraps=pygame.draw.line) as line, \
                                patch.dict(self.ui.fonts, {12: font}):
                            self.ui.draw_pointer_preview()
                        resolve.assert_called_once_with(button)
                        self.assertEqual(circle.call_args_list[0].args[2], expected_center)
                        self.assertEqual(self.ui.window.get_clip(), old_clip)
                        if expected == "damage":
                            self.assertEqual(circle.call_args_list[0].args[3],
                                             max(3, round(self.ui.radius * scale)))
                            line.assert_not_called()
                            font.render.assert_not_called()
                        else:
                            self.assertGreaterEqual(line.call_count, 2)
                            self.assertEqual(font.render.call_args.args[0],
                                             "L" if expected == "plant_left" else "R")
                        with patch.object(self.ui, "effective_pointer_tool", wraps=self.ui.effective_pointer_tool) as resolve, \
                                patch.object(self.ui, "interact") as interact:
                            self.click(pos, button)
                        resolve.assert_called_once_with(button)
                        expected_args = (*expected_cell,) if expected == "damage" else (
                            *expected_cell, 0 if expected == "plant_left" else 1)
                        self.assertEqual(interact.call_args.args, expected_args)
                        self.assertEqual(self.ui.pointer_tool, tool)
                        self.assert_frozen(before)

    def test_plant_marker_label_clears_top_captions_at_both_corners(self):
        for width, height in ((860, 740), (1100, 902)):
            self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
            caption_bottom = self.ui.board.top + 11 + self.ui.fonts[11].get_height()
            for tool in ("plant_left", "plant_right"):
                self.ui.action(("tool", tool))
                for x in (0, self.world.size - 1):
                    with self.subTest(width=width, tool=tool, x=x), \
                            patch("pygame.mouse.get_pos", return_value=self.cell(x, 0)), \
                            patch("pygame.mouse.get_pressed", return_value=(False, False, False)), \
                            patch("pygame.draw.rect", wraps=pygame.draw.rect) as rectangle:
                        self.ui.draw_pointer_preview()
                        backdrop = rectangle.call_args.args[2]
                        self.assertGreater(backdrop.top, caption_bottom)
                        self.assertTrue(self.ui.board.contains(backdrop))

    def assert_pointer_preview_and_click(self, expected_tool):
        """Check both consumers at this exact point in the input-event queue."""
        self.assertEqual(self.ui.effective_pointer_tool(), expected_tool)
        font = Mock(wraps=self.ui.fonts[12])
        with patch("pygame.mouse.get_pos", return_value=self.cell(3, 4)), \
                patch("pygame.mouse.get_pressed", return_value=(False, False, False)), \
                patch.dict(self.ui.fonts, {12: font}), \
                patch("pygame.draw.line", wraps=pygame.draw.line) as line, \
                patch("pygame.draw.circle", wraps=pygame.draw.circle) as circle, \
                patch.object(self.ui, "interact") as interact:
            self.ui.draw_pointer_preview()
            self.click(self.cell(3, 4))
        if expected_tool == "damage":
            interact.assert_called_once_with(3, 4)
            circle.assert_called()
            line.assert_not_called()
            font.render.assert_not_called()
        else:
            side = 0 if expected_tool == "plant_left" else 1
            interact.assert_called_once_with(3, 4, side)
            self.assertGreaterEqual(line.call_count, 2)
            self.assertEqual(font.render.call_args.args[0], "L" if side == 0 else "R")

    def test_physical_modifiers_cannot_reorder_click_press_click_release_click(self):
        for tool in TOOLS:
            for shift in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                for physical_mods in (0, pygame.KMOD_SHIFT):
                    with self.subTest(tool=tool, shift=shift, physical_mods=physical_mods):
                        self.ui.action(("tool", tool))
                        self.mods.return_value = physical_mods
                        self.mods.reset_mock()
                        before = self.snapshot()
                        # One SDL batch can contain all five events although
                        # get_mods already reflects a later physical state.
                        self.assert_pointer_preview_and_click(tool)
                        self.key(shift, mod=0)
                        self.assertEqual(self.ui.held_shift_keys, {shift})
                        self.assert_pointer_preview_and_click("plant_left")
                        self.key(shift, up=True, mod=pygame.KMOD_SHIFT)
                        self.assertEqual(self.ui.held_shift_keys, set())
                        self.assert_pointer_preview_and_click(tool)
                        self.assertEqual(self.ui.pointer_tool, tool)
                        self.mods.assert_not_called()
                        self.assert_frozen(before)

    def test_both_shift_keys_track_exact_releases_despite_opposing_physical_modifiers(self):
        for tool in TOOLS:
            for released, retained in ((pygame.K_LSHIFT, pygame.K_RSHIFT),
                                       (pygame.K_RSHIFT, pygame.K_LSHIFT)):
                with self.subTest(tool=tool, released=released):
                    self.ui.action(("tool", tool))
                    self.mods.return_value = 0
                    self.mods.reset_mock()
                    before = self.snapshot()
                    for shift in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                        for _ in range(3):
                            self.key(shift, repeat=True)
                    self.assertEqual(self.ui.held_shift_keys, {pygame.K_LSHIFT, pygame.K_RSHIFT})
                    self.assert_pointer_preview_and_click("plant_left")
                    self.key(released, up=True)
                    self.assertEqual(self.ui.held_shift_keys, {retained})
                    self.assert_pointer_preview_and_click("plant_left")
                    self.mods.return_value = pygame.KMOD_SHIFT
                    self.key(retained, up=True, mod=pygame.KMOD_SHIFT)
                    self.assertEqual(self.ui.held_shift_keys, set())
                    self.assert_pointer_preview_and_click(tool)
                    # A duplicate release must not rehydrate physical state.
                    self.key(retained, up=True, mod=pygame.KMOD_SHIFT)
                    self.assert_pointer_preview_and_click(tool)
                    self.assertEqual(self.ui.pointer_tool, tool)
                    self.mods.assert_not_called()
                    self.assert_frozen(before)

    def test_focus_loss_and_gain_require_a_new_shift_event_before_overriding_tool(self):
        for tool in TOOLS:
            for physical_mods in (0, pygame.KMOD_SHIFT):
                with self.subTest(tool=tool, physical_mods=physical_mods):
                    self.ui.action(("tool", tool))
                    self.mods.return_value = physical_mods
                    self.mods.reset_mock()
                    before = self.snapshot()
                    self.key(pygame.K_LSHIFT)
                    self.key(pygame.K_RSHIFT)
                    self.ui.event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
                    self.assertEqual(self.ui.held_shift_keys, set())
                    self.assert_pointer_preview_and_click(tool)
                    self.ui.event(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
                    self.assertEqual(self.ui.held_shift_keys, set())
                    self.assert_pointer_preview_and_click(tool)
                    self.key(pygame.K_LSHIFT, up=True, mod=pygame.KMOD_SHIFT)
                    self.assert_pointer_preview_and_click(tool)
                    self.key(pygame.K_RSHIFT, mod=0)
                    self.assertEqual(self.ui.held_shift_keys, {pygame.K_RSHIFT})
                    self.assert_pointer_preview_and_click("plant_left")
                    self.key(pygame.K_RSHIFT, up=True, mod=pygame.KMOD_SHIFT)
                    self.assert_pointer_preview_and_click(tool)
                    self.assertEqual(self.ui.pointer_tool, tool)
                    self.mods.assert_not_called()
                    self.assert_frozen(before)

    def test_initial_modifiers_are_empty_even_when_physical_shift_is_already_pressed(self):
        self.mods.return_value = pygame.KMOD_SHIFT
        self.mods.reset_mock()
        self.new_world()
        self.assertEqual(self.ui.held_shift_keys, set())
        self.assert_pointer_preview_and_click("damage")
        self.ui.action(("tool", "plant_right"))
        self.assert_pointer_preview_and_click("plant_right")
        self.mods.assert_not_called()

    def test_preview_is_static_with_fx_off_and_cannot_paint_outside_field(self):
        self.ui.reduced_motion = True
        self.ui.radius = self.world.size
        before = self.snapshot()
        for tool in TOOLS:
            for pos in (self.cell(3, 4), self.ui.board.topleft,
                        (self.ui.board.right - 1, self.ui.board.bottom - 1)):
                with self.subTest(tool=tool, pos=pos):
                    self.ui.action(("tool", tool))
                    images = []
                    for elapsed in (0, 7.5):
                        self.ui.elapsed = elapsed
                        self.ui.window.fill((3, 7, 11))
                        baseline = pygame.surfarray.array3d(self.ui.window)
                        with patch("pygame.mouse.get_pos", return_value=pos), \
                                patch("pygame.mouse.get_pressed", return_value=(False, False, False)):
                            self.ui.draw_pointer_preview()
                        pixels = pygame.surfarray.array3d(self.ui.window)
                        images.append(pixels)
                        outside = np.ones(pixels.shape[:2], dtype=bool)
                        outside[self.ui.board.left:self.ui.board.right,
                                self.ui.board.top:self.ui.board.bottom] = False
                        np.testing.assert_array_equal(pixels[outside], baseline[outside])
                        self.assertTrue(np.any(pixels != baseline))
                    np.testing.assert_array_equal(images[0], images[1])
                    self.assert_frozen(before)

    def test_pointer_preview_is_absent_outside_field_and_in_duel_or_lesson(self):
        for mode in ("lab", "duel", "lesson"):
            self.new_world()
            self.ui.action(("tool", "plant_left"))
            if mode != "lab":
                self.ui.action(mode)
            for pos in ((20, 15), (self.ui.rail_x + 20, 630), self.ui.board.center):
                if mode == "lab" and pos == self.ui.board.center:
                    continue
                with self.subTest(mode=mode, pos=pos), \
                        patch("pygame.mouse.get_pos", return_value=pos), \
                        patch("pygame.draw.circle") as circle, patch("pygame.draw.line") as line:
                    self.ui.draw_pointer_preview()
                    circle.assert_not_called()
                    line.assert_not_called()

    def test_lower_rail_labels_are_not_clipped_even_at_maximum_radius(self):
        for size in ((860, 740), (1100, 902)):
            self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=size[0], h=size[1]))
            for mode in ("hard", "soft"):
                self.world.mode = mode
                for radius in (1, 16, 256):
                    self.ui.radius = radius
                    # Isolate typography at the largest supported grid without
                    # allocating or stepping a full 256-square NCA fixture.
                    with self.subTest(size=size, mode=mode, radius=radius), \
                            patch.object(self.world, "size", max(16, radius)), \
                            patch.object(self.ui, "text", wraps=self.ui.text) as text:
                        self.ui.draw(dt=0)
                    labels = []
                    for call in text.call_args_list:
                        value, x, y = call.args[:3]
                        if x < self.ui.rail_x or y < 622:
                            continue
                        font_size = call.args[3] if len(call.args) > 3 else call.kwargs.get("size", 14)
                        width = call.args[5] if len(call.args) > 5 else call.kwargs.get("width")
                        measured = self.ui.fonts[font_size].size(value)
                        if width is not None:
                            self.assertLessEqual(measured[0], width, value)
                        bounds = pygame.Rect((x, y), measured)
                        self.assertTrue(self.ui.window.get_rect().contains(bounds), value)
                        for rect, action in self.ui.buttons:
                            self.assertFalse(bounds.colliderect(rect), (value, action))
                        labels.append(value)
                    self.assertIn(f"CUT RADIUS {radius}", labels)
                    self.assertIn("LEFT-CLICK TOOL", labels)

    def test_lab_actions_and_stale_hitboxes_are_inert_in_every_duel_phase(self):
        for mode in ("hard", "soft"):
            for phase in ("warmup", "running", "finished", "hidden", "invalid"):
                with self.subTest(mode=mode, phase=phase):
                    self.new_world("--mode", mode)
                    self.ui.action(("tool", "plant_right"))
                    lab_buttons = [(rect.copy(), action) for rect, action in self.ui.buttons]
                    self.ui.action("duel")
                    if phase == "running":
                        self.world.step(3)
                    elif phase in ("finished", "hidden"):
                        self.world.step(100)
                    elif phase == "invalid":
                        with torch.no_grad():
                            self.world.left["model"].fc1.bias[3] = float("inf")
                        self.world.step(1)
                    with np.errstate(invalid="ignore"):
                        self.ui.draw(dt=0)
                    self.assertFalse(any(action in LAB_ACTIONS for _, action in self.ui.buttons))
                    if phase == "hidden":
                        self.ui.action("result")
                        self.ui.draw(dt=0)
                        self.assertFalse(self.ui.result_visible)
                    # Deliberately retain the old lab hitboxes across the mode
                    # switch, as happens before the next render in a batch.
                    self.ui.buttons = lab_buttons
                    before = self.snapshot()
                    original = self.ui.pointer_tool, self.ui.radius, self.ui.paused
                    for action in LAB_ACTIONS:
                        self.ui.action(action)
                        self.click_control(action)
                        self.assert_frozen(before)
                        self.assertEqual((self.ui.pointer_tool, self.ui.radius, self.ui.paused), original)
                    for side in (None, 0, 1):
                        self.ui.interact(2, 2, side)
                    for button in (1, 3):
                        self.click(self.cell(2, 2), button)
                    self.key(pygame.K_LSHIFT)
                    self.click(self.cell(2, 2))
                    self.key(pygame.K_LSHIFT, up=True)
                    self.key(pygame.K_c)
                    self.assert_frozen(before)
                    self.assertEqual((self.ui.pointer_tool, self.ui.radius, self.ui.paused), original)

    def test_lab_actions_and_stale_hitboxes_are_inert_in_every_lesson_phase(self):
        for phase in LESSON_PHASES:
            with self.subTest(phase=phase):
                self.new_world()
                self.ui.action(("tool", "plant_left"))
                lab_buttons = [(rect.copy(), action) for rect, action in self.ui.buttons]
                self.reach_lesson(phase)
                self.assertFalse(any(action in LAB_ACTIONS for _, action in self.ui.buttons))
                self.ui.buttons = lab_buttons
                before = self.snapshot()
                original = self.ui.pointer_tool, self.ui.radius, self.ui.paused
                for action in LAB_ACTIONS:
                    self.ui.action(action)
                    self.click_control(action)
                    self.assert_frozen(before)
                    self.assertEqual((self.ui.pointer_tool, self.ui.radius, self.ui.paused), original)
                for side in (None, 0, 1):
                    self.ui.interact(2, 2, side)
                for button in (1, 3):
                    self.click(self.cell(2, 2), button)
                self.key(pygame.K_LSHIFT)
                self.click(self.cell(2, 2))
                self.key(pygame.K_LSHIFT, up=True)
                for key in (pygame.K_c, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    self.key(key)
                self.assert_frozen(before)
                self.assertEqual((self.ui.pointer_tool, self.ui.radius, self.ui.paused), original)

    def test_persistent_tool_cannot_bypass_lesson_seed_or_change_cut_click_gate(self):
        for tool in TOOLS:
            with self.subTest(tool=tool):
                self.new_world()
                self.ui.action(("tool", tool))
                self.reach_lesson("seed")
                center = self.world.size // 2
                before = self.snapshot()
                self.click(self.cell(center, center))
                self.click(self.cell(center, center), button=3)
                self.assert_frozen(before)
                self.assertEqual(self.world.lesson.phase, "seed")
                self.key(pygame.K_LSHIFT)
                self.click(self.cell(center, center))
                self.key(pygame.K_LSHIFT, up=True)
                self.assertEqual(self.world.lesson.phase, "grow")
                self.assertEqual(self.world.steps, 0)
                seeded = self.snapshot()
                self.click(self.cell(center, center))
                self.assert_frozen(seeded)
                self.reach_lesson("damage")
                plan = self.world.lesson.snapshot()["plan"]
                before = self.snapshot()
                self.key(pygame.K_LSHIFT)
                self.click(self.cell(plan["x"], plan["y"]))
                self.key(pygame.K_LSHIFT, up=True)
                self.click(self.cell(plan["x"], plan["y"]), button=3)
                self.assert_frozen(before)
                self.click(self.cell(plan["x"], plan["y"]))
                self.assertEqual(self.world.lesson.phase, "injured")
                self.assertTrue(self.ui.paused)
                self.assertEqual(self.world.stats()["left_alive"], plan["remaining"])
                injured = self.snapshot()
                for _ in range(3):
                    self.click(self.cell(plan["x"], plan["y"]))
                self.assert_frozen(injured)

    def test_lesson_seed_gate_uses_only_modifier_events_before_each_queued_click(self):
        scenarios = {
            "press_after_first_click": (("click", None), ("down", None), ("click", 0)),
            "press_click_release_click": (("down", None), ("click", 0), ("up", None), ("click", None)),
            "press_release_then_click": (("down", None), ("up", None), ("click", None)),
            "focus_reentry_before_press": (("down", None), ("lost", None), ("gained", None),
                                           ("click", None), ("down", None), ("click", 0)),
        }
        for tool in TOOLS:
            for shift in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                for physical_mods in (0, pygame.KMOD_SHIFT):
                    for name, events in scenarios.items():
                        with self.subTest(tool=tool, shift=shift, physical_mods=physical_mods, events=name):
                            self.mods.return_value = physical_mods
                            self.new_world()
                            self.ui.action(("tool", tool))
                            self.reach_lesson("seed")
                            self.mods.reset_mock()
                            center = self.world.size // 2
                            before = self.snapshot()
                            planted = False
                            with patch.object(self.ui, "interact", wraps=self.ui.interact) as interact:
                                for kind, side in events:
                                    if kind == "down":
                                        self.key(shift, mod=0)
                                    elif kind == "up":
                                        self.key(shift, up=True, mod=pygame.KMOD_SHIFT)
                                    elif kind in ("lost", "gained"):
                                        self.ui.event(pygame.event.Event(
                                            pygame.WINDOWFOCUSLOST if kind == "lost" else pygame.WINDOWFOCUSGAINED))
                                    else:
                                        self.assertEqual(self.ui.effective_pointer_tool(),
                                                         "damage" if side is None else "plant_left")
                                        self.click(self.cell(center, center))
                                        expected = (center, center) if side is None else (center, center, side)
                                        self.assertEqual(interact.call_args.args, expected)
                                        if side == 0 and not planted:
                                            planted = True
                                            self.assertEqual(self.world.lesson.phase, "grow")
                                            self.assertEqual(self.world.a[0, 3:5, center, center].tolist(), [1, 1])
                                            before = self.snapshot()
                                        else:
                                            self.assert_frozen(before)
                                        self.assertEqual(self.world.lesson.phase, "grow" if planted else "seed")
                                        self.assertEqual(self.world.steps, 0)
                            self.assertEqual(self.ui.pointer_tool, tool)
                            self.mods.assert_not_called()

    def test_existing_bracket_shortcut_still_changes_duel_brush_without_editing(self):
        self.ui.action(("tool", "plant_left"))
        self.ui.action("duel")
        before = self.snapshot()
        original_radius = self.ui.radius
        self.key(pygame.K_RIGHTBRACKET)
        self.assertEqual(self.ui.radius, min(self.world.size, original_radius + 1))
        self.key(pygame.K_LEFTBRACKET)
        self.assertEqual(self.ui.radius, original_radius)
        self.assertEqual(self.ui.pointer_tool, "plant_left")
        self.assert_frozen(before)

    def test_mouse_radius_changes_apply_the_displayed_inclusive_crater(self):
        for radius, removed in ((1, 2), (2, 4)):
            with self.subTest(radius=radius):
                self.click_control("clear")
                while self.ui.radius > 1:
                    self.click_control(("radius", -1))
                if radius == 2:
                    self.click_control(("radius", 1))
                self.click_control(("tool", "plant_left"))
                for x, y in ((8, 8), (9, 8), (10, 8), (11, 8), (8, 10)):
                    self.click(self.cell(x, y))
                self.click_control(("tool", "plant_right"))
                self.click(self.cell(8, 9))
                self.assert_scores(5, 1)
                self.click_control(("tool", "damage"))
                self.click(self.cell(8, 8))
                self.assert_scores(5 - removed, 0)
                self.assertEqual(self.ui.effects[-1]["radius"], radius)
                self.assertIn(f"L -{removed} / R -1 life", self.ui.effects[-1]["label"])
                self.assertEqual(self.world.a[0, 3, 8, 11].item(), 1)
                self.assertEqual(self.world.steps, 0)

    def test_selected_control_is_unique_and_tracks_tool_without_changing_picker_side(self):
        for tool in TOOLS:
            self.click_control("right")
            self.click_control(("tool", tool))
            with patch.object(self.ui, "button", wraps=self.ui.button) as button:
                self.ui.draw(dt=0)
            active_tools = [call.args[2][1] for call in button.call_args_list
                            if call.args[2] in TOOL_ACTIONS and call.args[3]]
            self.assertEqual(active_tools, [tool])
            self.assertEqual(self.ui.side, 1)

    def test_number_key_culture_shortcuts_preserve_tool_and_side_specific_pins(self):
        self.ui.action(("tool", "plant_right"))
        for key, mod, side in ((pygame.K_3, 0, 0),
                               (pygame.K_6, pygame.KMOD_SHIFT, 1)):
            self.key(key, mod=mod)
            self.assertEqual(self.ui.side, side)
            self.assertEqual(self.ui.pointer_tool, "plant_right")
            self.assertEqual((self.world.args.left_seed, self.world.args.right_seed), (0, 1))
            bundle = self.world.left if side == 0 else self.world.right
            self.assertEqual(bundle["seed_dir"], f"seed_{side:03d}")
            self.assertEqual(self.world.steps, 0)

    def test_direct_duel_and_lesson_startup_return_to_default_damage_in_lab(self):
        for mode in ("duel", "lesson"):
            with self.subTest(mode=mode):
                self.new_world("--" + mode)
                self.assertEqual(self.ui.pointer_tool, "damage")
                self.ui.action(mode)
                self.assertIsNone(self.world.lesson)
                self.assertIsNone(self.world.duel)
                self.assertEqual(self.ui.pointer_tool, "damage")
                with patch.object(self.ui, "interact") as interact:
                    self.click(self.cell(3, 4))
                interact.assert_called_once_with(3, 4)

    def test_initial_radius_keeps_default_and_caps_explicit_oversized_lab_brush(self):
        self.assertEqual(arena.parse_args([]).crater_radius, 4)
        self.assertEqual(self.ui.radius, 4)
        for size in (8, 48):
            with self.subTest(size=size):
                self.new_world("--grid-size", str(size), "--crater-radius", "80")
                self.assertEqual(self.ui.radius, size)
                self.assertEqual(self.world.args.crater_radius, 80)
                self.assertEqual(self.ui.pointer_tool, "damage")
                self.assertFalse(self.ui.paused)
                self.assertIn(str(size), self.ui.message)
                self.assertTrue(any(str(size) in notice[0] for notice in self.ui.notices))
                before = self.snapshot()
                self.click_control(("radius", 1))
                self.assertEqual(self.ui.radius, size)
                self.click_control(("radius", -1))
                self.assertEqual(self.ui.radius, size - 1)
                self.assert_frozen(before)

    def test_smaller_grid_reset_clamps_lab_radius_without_changing_tool_or_pause(self):
        for action in ("reset", "mode", ("target", 3)):
            with self.subTest(action=action):
                self.new_world("--grid-size", "32", "--crater-radius", "30")
                self.ui.action(("tool", "plant_right"))
                self.ui.paused = True
                self.world.args.grid_size = 8
                self.ui.action(action)
                self.assertEqual(self.world.size, 8)
                self.assertEqual(self.ui.radius, 8)
                self.assertEqual(self.ui.pointer_tool, "plant_right")
                self.assertTrue(self.ui.paused)
                self.assertEqual(self.world.args.crater_radius, 30)
                self.click_control(("radius", 1))
                self.assertEqual(self.ui.radius, 8)
                self.ui.radius = 0
                self.ui.refresh()
                self.assertEqual(self.ui.radius, 1)
                self.assertEqual(self.ui.pointer_tool, "plant_right")
                self.assertTrue(self.ui.paused)

    def test_inactive_duel_and_lesson_radius_only_caps_on_return_to_lab(self):
        for mode in ("duel", "lesson"):
            with self.subTest(mode=mode):
                self.new_world("--" + mode, "--crater-radius", "80")
                self.assertEqual(self.ui.radius, 80)
                self.ui.refresh()
                self.assertEqual(self.ui.radius, 80)
                before = self.snapshot()
                self.ui.action(("radius", -1))
                self.assertEqual(self.ui.radius, 80)
                self.assert_frozen(before)
                self.ui.action(mode)
                self.assertEqual(self.ui.radius, self.world.size)
                self.assertEqual(self.world.size, 16)
                self.assertEqual(self.world.args.crater_radius, 80)
                self.assertEqual(self.ui.pointer_tool, "damage")

    def test_direct_arena_damage_retains_zero_and_oversized_radius_contract(self):
        self.world.clear()
        self.world.plant(4, 4, 0)
        self.world.plant(5, 4, 0)
        with patch("arena.crater", wraps=arena.crater) as crater:
            self.world.damage(4, 4, 0)
            self.assertEqual(crater.call_args.args[-1], 0)
        self.assertEqual(self.world.stats()["left_alive"], 1)
        self.assertEqual(self.world.a[0, 3, 4, 5].item(), 1)
        self.world.plant(0, 0, 0)
        self.world.plant(15, 15, 1)
        before_rng = self.random_states()
        with patch("arena.crater", wraps=arena.crater) as crater:
            self.world.damage(0, 0, 80)
            self.assertEqual(crater.call_args.args[-1], 80)
        self.assertEqual((self.world.stats()["left_alive"], self.world.stats()["right_alive"]), (0, 0))
        self.assertEqual(self.world.steps, 0)
        self.assertEqual(self.ui.radius, 4)
        self.assert_rng_equal(before_rng)

    def test_unknown_tool_value_is_ignored_without_falling_back_to_culture_selection(self):
        self.ui.action(("tool", "plant_right"))
        before = self.snapshot()
        with patch.object(self.world, "select", side_effect=AssertionError("unknown tool became target")):
            self.ui.action(("tool", "not_a_tool"))
        self.assertEqual(self.ui.pointer_tool, "plant_right")
        self.assert_frozen(before)


if __name__ == "__main__":
    unittest.main()
