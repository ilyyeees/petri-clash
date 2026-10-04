"""CPU and dummy-SDL integration contracts for the optional regrowth lesson.

The fast model expands/repairs a bounded patch and evolves stochastic hidden
channels. It is deliberately not an NCA quality test: exact tensor/RNG replay,
gates, edits, pacing and presentation are the subjects. Shipped checkpoint
smokes are bounded, CPU-only and opt-in with PETRI_TEST_CHECKPOINTS=1.
"""

import contextlib
import io
import json
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
import torch.nn.functional as F

import arena
import clash
from lesson import LessonConfig


class PatchModel(torch.nn.Module):
    """State-driven 11x11 growth, a deterministic repair front, random memory."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.repair = True
        self.nonfinite = None

    def forward(self, state):
        self.calls += 1
        life = state[:, 3:4] > .1
        if self.repair:
            life = F.max_pool2d(life.float(), 3, stride=1, padding=1) > 0
        h, w = state.shape[-2:]
        window = torch.zeros_like(life)
        window[:, :, h // 2 - 5:h // 2 + 6, w // 2 - 5:w // 2 + 6] = True
        life = (life & window).float()
        result = torch.zeros_like(state)
        result[:, :3] = life * .6
        result[:, 3:5] = life
        result[:, 5:] = (state[:, 5:] * .8 + torch.rand_like(state[:, 5:]) * .02) * life
        if self.nonfinite is not None:
            result[0, -1, h // 2, w // 2] = self.nonfinite
        return result


def synthetic_bundle(target, device, *args, **kwargs):
    return {"model": PatchModel().to(device).eval(), "channels": 8,
            "grid_size": 16, "channels_last": False, "name": target.stem,
            "health": "ready", "seed_dir": "seed_000", "score": .01,
            "kind": "synthetic-patch", "source": "lesson-integration-test"}


PHASES = ("seed", "grow", "damage", "injured", "recover", "complete",
          "timeout", "unavailable", "invalid")
SHORT = LessonConfig(grow_ticks=7, recovery_ticks=16, hold_ticks=3)


class LessonTestCase(unittest.TestCase):
    def setUp(self):
        self.torch_rng = torch.get_rng_state()
        self.numpy_rng = np.random.get_state()
        self.python_rng = random.getstate()
        self.threads = torch.get_num_threads()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.loader = patch("arena.ensure_model", side_effect=synthetic_bundle)
        self.loader.start()
        self.new_world()

    def tearDown(self):
        pygame.quit()
        self.loader.stop()
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(self.deterministic)
        torch.set_rng_state(self.torch_rng)
        np.random.set_state(self.numpy_rng)
        random.setstate(self.python_rng)

    def new_world(self, *extra):
        args = arena.parse_args(["--device", "cpu", "--cpu-threads", "1",
                                 "--grid-size", "16", "--seed", "27", *extra])
        self.world = arena.Arena(args, clash.list_targets())
        return self.world

    def tensors(self):
        return tuple(value.clone() for value in (self.world.a, self.world.b,
                                                self.world.owner, self.world.control))

    def assert_tensors_equal(self, expected):
        for actual, previous in zip(self.tensors(), expected):
            self.assertTrue(torch.equal(actual, previous))

    def random_states(self):
        return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()

    def assert_rng_equal(self, expected):
        current = self.random_states()
        self.assertTrue(torch.equal(current[0], expected[0]))
        self.assertEqual(current[1][0], expected[1][0])
        np.testing.assert_array_equal(current[1][1], expected[1][1])
        self.assertEqual(current[1][2:], expected[1][2:])
        self.assertEqual(current[2], expected[2])

    def reach(self, phase, config=SHORT):
        model = self.world.left["model"]
        model.repair, model.nonfinite = True, None
        self.world.start_lesson(config)
        if phase != "seed":
            self.world.plant_lesson()
        if phase == "grow":
            self.world.step(1)
        elif phase == "unavailable":
            model.repair = False
            self.world.step(10_000)
        elif phase == "invalid":
            model.nonfinite = float("nan")
            self.world.step(10_000)
        elif phase not in ("seed", "grow"):
            self.world.step(10_000)
            if phase != "damage":
                self.world.cut_lesson()
            if phase not in ("damage", "injured"):
                self.world.watch_lesson()
                if phase == "recover":
                    self.world.step(1)
                elif phase == "timeout":
                    model.repair = False
                    self.world.step(10_000)
                else:
                    self.world.step(10_000)
        self.assertEqual(self.world.lesson.phase, phase)
        if hasattr(self, "ui"):
            self.ui.refresh(reset=True)
            self.ui.draw(dt=0)

    def assert_seed(self):
        self.assertEqual(self.world.lesson.phase, "seed")
        self.assertEqual(self.world.steps, 0)
        self.assertEqual((self.world.size, self.world.mode), (48, "soft"))
        self.assertFalse(self.world.duel_enabled)
        self.assertIsNone(self.world.duel)
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in self.tensors()))


class LessonArenaIntegrationTests(LessonTestCase):
    def test_direct_lesson_loads_only_its_culture_then_resolves_the_opponent_on_exit(self):
        with patch("arena.ensure_model", side_effect=synthetic_bundle) as load:
            self.new_world("--lesson", "--right", "7")
            self.assertEqual([call.args[0].stem for call in load.call_args_list], ["01_heart"])
            self.assertIsNone(self.world.report(1)["right"])
            self.assertIsNone(self.world.stop_lesson())
            self.assertEqual([call.args[0].stem for call in load.call_args_list], ["01_heart", "07_umbrella"])
        self.assertIsNone(self.world.lesson)
        self.assertEqual(self.world.right["name"], "07_umbrella")
        self.assertEqual((self.world.mode, self.world.size), ("hard", 16))

    def test_unavailable_deferred_opponent_has_an_explicit_loaded_culture_fallback(self):
        def load(target, *args, **kwargs):
            if target.stem == "04_moon":
                raise ValueError("collapsed checkpoint")
            return synthetic_bundle(target, *args, **kwargs)
        with patch("arena.ensure_model", side_effect=load):
            self.new_world("--lesson", "--right", "4")
            self.world.plant_lesson()
            self.world.step(4)
            warning = self.world.stop_lesson()
        self.assertIn("Right unavailable", warning)
        self.assertIn("using heart seed 000 ready", warning)
        self.assertIn("Auto selection kept", warning)
        self.assertIsNone(self.world.lesson)
        self.assertIs(self.world.left, self.world.right)
        self.assertEqual(self.world.left_index, self.world.right_index)
        self.assertEqual((self.world.stats()["left_alive"], self.world.stats()["right_alive"]), (1, 1))

    def test_start_forces_fresh_solo_field_and_restores_original_settings(self):
        for mode, grid in (("hard", 0), ("hard", 23), ("soft", 16)):
            with self.subTest(mode=mode, grid=grid):
                self.new_world("--mode", mode, "--grid-size", str(grid), "--duel")
                self.world.start_lesson()
                self.assert_seed()
                self.assertEqual(self.world.start_positions, {"left": [24, 24], "right": None})
                self.world.plant_lesson()
                self.world.step(8)
                self.world.start_lesson()
                self.assert_seed()
                self.world.stop_lesson()
                self.assertIsNone(self.world.lesson)
                self.assertEqual((self.world.mode, self.world.args.grid_size), (mode, grid))
                self.assertEqual(self.world.size, grid or 16)
                self.assertIsNone(self.world.duel)
                self.assertFalse(self.world.duel_enabled)
                self.assertEqual(self.world.steps, 0)
                self.assertEqual(self.world.stats()["left_alive"], 1)
                self.assertEqual(self.world.stats()["right_alive"], 1)

    def test_center_seed_and_every_paused_gate_reject_extra_ticks(self):
        self.world.start_lesson()
        self.assertEqual(self.world.step(1_000), 0)
        self.world.plant_lesson()
        self.assertEqual(self.world.a[0, 3:5, 24, 24].tolist(), [1, 1])
        self.assertEqual(torch.count_nonzero(self.world.a).item(), 2)
        self.assertEqual(self.world.steps, 0)
        self.world.step(1_000)
        self.assertEqual(self.world.lesson.phase, "damage")
        self.assertEqual(self.world.steps, 160)
        self.world.cut_lesson()
        self.assertEqual(self.world.lesson.phase, "injured")
        self.assertEqual(self.world.steps, 160)
        for phase in ("seed", "damage", "injured", "complete", "timeout", "unavailable", "invalid"):
            with self.subTest(phase=phase):
                self.reach(phase)
                expected, rng = self.tensors(), self.random_states()
                state = self.world.lesson.snapshot()
                self.assertEqual(self.world.step(10_000), 0)
                self.assertEqual(self.world.lesson.snapshot(), state)
                self.assert_tensors_equal(expected)
                self.assert_rng_equal(rng)

    def test_tick_by_tick_counts_and_bitwise_replay_are_speed_independent(self):
        trajectories = []
        for speed in (1, 2, 4, 8, 10_000):
            self.world.start_lesson()
            self.world.plant_lesson()
            counts = []
            original = self.world.lesson.record_tick

            def record(tick, living, **kwargs):
                counts.append((tick, living))
                return original(tick, living, **kwargs)

            with patch.object(self.world.lesson, "record_tick", side_effect=record):
                while self.world.lesson.phase == "grow":
                    self.world.step(speed)
                self.assertEqual(self.world.steps, 160)
                self.assertEqual(self.world.lesson.snapshot()["baseline"], 121)
                self.world.cut_lesson()
                self.world.watch_lesson()
                while self.world.lesson.can_step:
                    self.world.step(speed)
            self.assertEqual(self.world.lesson.phase, "complete")
            self.assertEqual([tick for tick, _ in counts], list(range(1, self.world.steps + 1)))
            self.assertEqual(self.world.lesson.snapshot()["hold"], 24)
            self.assertTrue(torch.count_nonzero(self.world.a[:, 5:]).item() > 0)
            trajectories.append((counts, self.world.lesson.snapshot(), self.tensors(), self.random_states()))
        first = trajectories[0]
        for counts, state, tensors, rng in trajectories[1:]:
            self.assertEqual(counts, first[0])
            self.assertEqual(state, first[1])
            for actual, expected in zip(tensors, first[2]):
                self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(rng[0], first[3][0]))
            np.testing.assert_array_equal(rng[1][1], first[3][1][1])
            self.assertEqual(rng[2], first[3][2])

    def test_only_visible_model_is_called_even_with_empty_opponent(self):
        self.world.start_lesson()
        with patch.object(self.world.right["model"], "forward", side_effect=AssertionError("invisible opponent advanced")):
            self.world.plant_lesson()
            self.world.step(999)
            self.world.cut_lesson()
            self.world.watch_lesson()
            self.world.step(999)
        self.assertEqual(self.world.left["model"].calls, self.world.steps)
        self.assertEqual(torch.count_nonzero(self.world.b).item(), 0)
        self.assertEqual(self.world.stats()["right_alive"], 0)

    def test_exact_preview_removes_all_channels_once_without_consuming_rng_or_ticks(self):
        self.reach("damage")
        before, rng = self.tensors(), self.random_states()
        state = self.world.lesson.snapshot()
        plan = state["plan"]
        yy, xx = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
        mask = (xx - plan["x"]) ** 2 + (yy - plan["y"]) ** 2 <= plan["radius"] ** 2
        removed = int(((before[0][0, 3] > .1) & mask).sum())
        self.assertEqual(removed, plan["removed"])
        self.assertGreater(torch.count_nonzero(before[0][0, 5:, mask]).item(), 0)
        self.world.cut_lesson()
        self.assertEqual(self.world.steps, SHORT.grow_ticks)
        self.assertEqual(self.world.lesson.snapshot()["remaining_after_cut"], plan["remaining"])
        self.assertEqual(self.world.stats()["left_alive"], plan["baseline"] - removed)
        for actual, original in zip(self.tensors(), before):
            self.assertEqual(torch.count_nonzero(actual[0, :, mask]).item(), 0)
            self.assertTrue(torch.equal(actual[0, :, ~mask], original[0, :, ~mask]))
        self.assert_rng_equal(rng)
        injured = self.tensors()
        with self.assertRaises(ValueError):
            self.world.cut_lesson()
        self.assert_tensors_equal(injured)
        self.assertEqual(self.world.lesson.phase, "injured")

    def test_nonfinite_output_quarantines_visible_and_hidden_channels_at_exact_tick(self):
        for phase in ("grow", "recover"):
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(phase=phase, nonfinite=nonfinite):
                    self.reach(phase)
                    before, tick = self.tensors(), self.world.steps
                    self.world.left["model"].nonfinite = nonfinite
                    self.assertEqual(self.world.step(999), 1)
                    self.assertEqual(self.world.steps, tick + 1)
                    self.assertEqual(self.world.lesson.phase, "invalid")
                    self.assertFalse(self.world.lesson.snapshot()["valid"])
                    self.assertIsNone(self.world.lesson.snapshot()["current"])
                    self.assert_tensors_equal(before)
                    self.assertTrue(self.world.stats()["finite"])
                    self.assertEqual(self.world.step(999), 0)

    def test_recovery_timeout_is_exact_and_unavailable_does_not_offer_a_cut(self):
        self.world.start_lesson()
        self.world.plant_lesson()
        self.world.step(1_000)
        self.world.cut_lesson()
        self.world.watch_lesson()
        self.world.left["model"].repair = False
        self.assertEqual(self.world.step(1_000), 160)
        self.assertEqual(self.world.steps, 320)
        self.assertEqual(self.world.lesson.phase, "timeout")
        self.assertEqual(self.world.lesson.snapshot()["hold"], 0)
        self.reach("unavailable")
        self.assertIsNone(self.world.lesson.snapshot()["plan"])
        with self.assertRaises(ValueError):
            self.world.cut_lesson()

    def test_direct_free_edits_and_wrong_side_are_atomic_in_every_phase(self):
        for phase in PHASES:
            self.reach(phase)
            before, rng, state = self.tensors(), self.random_states(), self.world.lesson.snapshot()
            for operation in (lambda: self.world.plant(2, 2, 0), lambda: self.world.plant(2, 2, 1),
                              lambda: self.world.damage(24, 24, 40), self.world.clear,
                              lambda: self.world.select(1, 1)):
                with self.subTest(phase=phase, operation=operation), self.assertRaises(ValueError):
                    operation()
                self.assert_tensors_equal(before)
                self.assert_rng_equal(rng)
                self.assertEqual(self.world.lesson.snapshot(), state)

    def test_reset_selection_and_duel_interrupt_every_phase_cleanly(self):
        for phase in PHASES:
            for action in ("reset", "select", "duel", "stop"):
                with self.subTest(phase=phase, action=action):
                    self.reach(phase)
                    if action == "reset":
                        self.world.reset()
                        self.assert_seed()
                        self.assertEqual(self.world.lesson.config, SHORT)
                    elif action == "select":
                        self.world.select(5, 0)
                        self.assertEqual(self.world.left_index, 5)
                        self.assert_seed()
                    elif action == "duel":
                        self.world.set_duel(True)
                        self.assertIsNone(self.world.lesson)
                        self.assertTrue(self.world.duel_enabled)
                        self.assertEqual(self.world.duel.tick, 0)
                        self.assertEqual((self.world.mode, self.world.size), ("hard", 16))
                    else:
                        self.world.stop_lesson()
                        self.assertIsNone(self.world.lesson)
                        self.assertIsNone(self.world.duel)
                        self.assertEqual((self.world.mode, self.world.size), ("hard", 16))
                    self.assertEqual(self.world.steps, 0)

    def test_failed_loading_preserves_phase_tensor_rng_and_selection_in_every_phase(self):
        def failed(*args, **kwargs):
            torch.rand(17)  # Failed construction must also preserve simulation RNG.
            raise ValueError("synthetic unavailable checkpoint")

        for phase in PHASES:
            with self.subTest(phase=phase):
                self.reach(phase)
                before, rng, state = self.tensors(), self.random_states(), self.world.lesson.snapshot()
                left, index = self.world.left, self.world.left_index
                with patch("arena.ensure_model", side_effect=failed), self.assertRaises(ValueError):
                    self.world.select(8, 0)
                self.assertIs(self.world.left, left)
                self.assertEqual(self.world.left_index, index)
                self.assertEqual(self.world.lesson.snapshot(), state)
                self.assert_tensors_equal(before)
                self.assert_rng_equal(rng)

    def test_reports_are_fresh_json_snapshots_without_aliasing_lesson(self):
        for phase in PHASES:
            with self.subTest(phase=phase):
                self.reach(phase)
                state = self.world.lesson.snapshot()
                report = self.world.report(1)
                self.assertEqual(report["lesson"], state)
                self.assertEqual(report["step"], state["tick"])
                self.assertEqual(report["placement"], "solo-center")
                self.assertEqual(report["starting_positions"], {"left": [24, 24], "right": None})
                self.assertIsNone(report["duel"])
                json.dumps(report, allow_nan=False)
                report["lesson"]["config"]["grow_ticks"] = -1
                if report["lesson"]["plan"]:
                    report["lesson"]["plan"]["removed"] = -1
                self.assertEqual(self.world.report(1)["lesson"], state)


class LessonUIIntegrationTests(LessonTestCase):
    def setUp(self):
        super().setUp()
        self.ui = arena.ArenaUI(self.world)
        self.ui.draw(dt=0)

    def key(self, key, mod=0):
        self.ui.event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))

    def click(self, pos, button=1, shift=False):
        if shift:
            self.ui.event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LSHIFT, mod=pygame.KMOD_LSHIFT))
        self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))
        if shift:
            self.ui.event(pygame.event.Event(pygame.KEYUP, key=pygame.K_LSHIFT, mod=0))

    def cell(self, x, y):
        return (self.ui.board.x + round((x + .5) * self.ui.board.width / self.world.size),
                self.ui.board.y + round((y + .5) * self.ui.board.height / self.world.size))

    def control(self, action):
        return next(rect for rect, value in self.ui.buttons if value == action)

    def test_keyboard_button_and_cli_enter_seed_stage_with_same_settings(self):
        for entry in ("keyboard", "button", "cli"):
            with self.subTest(entry=entry):
                if entry == "cli":
                    self.new_world("--lesson", "--steps-per-frame", "8")
                    self.ui = arena.ArenaUI(self.world)
                else:
                    self.ui.speed, self.ui.side, self.ui.view = 8, 1, "territory"
                    if entry == "keyboard":
                        self.key(pygame.K_g)
                    else:
                        self.click(self.control("lesson").center)
                self.ui.draw(dt=0)
                self.assert_seed()
                self.assertEqual((self.ui.side, self.ui.view, self.ui.speed), (0, "organisms", 1))
                self.assertTrue(self.ui.paused)
                self.assertEqual(self.ui.lesson_snapshot, self.world.lesson.snapshot())
                self.key(pygame.K_g)
                self.ui.draw(dt=0)
                self.assertIsNone(self.world.lesson)
                self.assertEqual(self.ui.speed, 8)
                self.assertEqual((self.world.mode, self.world.size), ("hard", 16))

    def test_enter_and_marked_click_have_identical_seed_and_damage_results(self):
        results = []
        for method in ("keyboard", "mouse"):
            self.reach("seed")
            if method == "keyboard":
                self.key(pygame.K_RETURN)
            else:
                self.click(self.cell(24, 24), shift=True)
            self.assertEqual(self.world.lesson.phase, "grow")
            self.ui.draw(dt=0)
            self.world.step(10_000)
            self.ui.draw(dt=0)
            self.assertEqual(self.world.lesson.phase, "damage")
            plan = self.world.lesson.snapshot()["plan"]
            if method == "keyboard":
                self.key(pygame.K_RETURN)
            else:
                self.click(self.cell(plan["x"], plan["y"]))
            self.assertEqual(self.world.lesson.phase, "injured")
            self.assertTrue(self.ui.paused)
            results.append((self.world.lesson.snapshot(), self.tensors(), self.random_states()))
        self.assertEqual(results[0][0], results[1][0])
        for expected, actual in zip(results[0][1], results[1][1]):
            self.assertTrue(torch.equal(expected, actual))
        self.assert_rng_equal(results[0][2])

    def test_unmarked_clicks_and_nonprimary_buttons_do_not_change_gates(self):
        for phase in ("seed", "damage", "injured", "complete"):
            self.reach(phase)
            before, rng, state = self.tensors(), self.random_states(), self.world.lesson.snapshot()
            for pos, button, shift in ((self.cell(1, 1), 1, False), (self.cell(1, 1), 1, True),
                                       (self.cell(24, 24), 3, False), (self.cell(24, 24), 2, False),
                                       (self.cell(24, 24), 4, False), (self.cell(24, 24), 5, False),
                                       ((10, 10), 1, False)):
                with self.subTest(phase=phase, button=button, shift=shift):
                    self.click(pos, button, shift)
                    self.assertEqual(self.world.lesson.snapshot(), state)
                    self.assert_tensors_equal(before)
                    self.assert_rng_equal(rng)

    def test_duplicate_primary_inputs_cannot_skip_the_paused_injury_frame(self):
        for second in ("enter", "click", "button"):
            with self.subTest(second=second):
                self.reach("damage")
                plan = self.world.lesson.snapshot()["plan"]
                primary = self.control("lesson_primary").center
                self.key(pygame.K_RETURN)
                state, before = self.world.lesson.snapshot(), self.tensors()
                if second == "enter":
                    self.key(pygame.K_RETURN)
                elif second == "click":
                    self.click(self.cell(plan["x"], plan["y"]))
                else:
                    self.click(primary)
                self.assertEqual(self.world.lesson.phase, "injured")
                self.assertEqual(self.world.lesson.snapshot(), state)
                self.assertTrue(self.ui.paused)
                self.assert_tensors_equal(before)
                self.ui.draw(dt=0)
                self.key(pygame.K_RETURN)
                self.assertEqual(self.world.lesson.phase, "recover")
                self.assertFalse(self.ui.paused)

    def test_primary_during_growth_and_recovery_is_a_nonmutating_noop(self):
        for phase in ("grow", "recover"):
            self.reach(phase)
            state, before, rng = self.world.lesson.snapshot(), self.tensors(), self.random_states()
            for _ in range(3):
                self.key(pygame.K_RETURN)
                self.ui.action("lesson_primary")
            self.assertEqual(self.world.lesson.snapshot(), state)
            self.assert_tensors_equal(before)
            self.assert_rng_equal(rng)

    def test_space_and_step_never_bypass_seed_or_damage_but_start_injured_watch(self):
        for phase in ("seed", "damage", "complete", "timeout", "unavailable", "invalid"):
            self.reach(phase)
            before, state = self.tensors(), self.world.lesson.snapshot()
            for key in (pygame.K_SPACE, pygame.K_n):
                self.key(key)
                self.assertEqual(self.world.lesson.snapshot(), state)
                self.assertTrue(self.ui.paused)
                self.assert_tensors_equal(before)
        for key in (pygame.K_SPACE, pygame.K_n):
            self.reach("injured")
            tick = self.world.steps
            self.key(key)
            self.assertEqual(self.world.lesson.phase, "recover")
            self.assertEqual(self.world.steps, tick + int(key == pygame.K_n))
            self.assertEqual(self.ui.paused, key == pygame.K_n)

    def test_terminal_enter_restarts_and_repeated_enter_does_not_plant_before_redraw(self):
        for phase in ("complete", "timeout", "unavailable", "invalid"):
            with self.subTest(phase=phase):
                self.reach(phase)
                self.key(pygame.K_RETURN)
                self.assert_seed()
                self.key(pygame.K_RETURN)
                self.assert_seed()
                self.ui.draw(dt=0)
                self.key(pygame.K_RETURN)
                self.assertEqual(self.world.lesson.phase, "grow")

    def test_edit_mode_radius_and_right_side_controls_are_locked_in_every_phase(self):
        for phase in PHASES:
            self.reach(phase)
            before, state, rng = self.tensors(), self.world.lesson.snapshot(), self.random_states()
            radius = self.ui.radius
            for key in (pygame.K_c, pygame.K_m, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                self.key(key)
                self.assertEqual(self.world.lesson.snapshot(), state)
                self.assertEqual(self.world.mode, "soft")
                self.assertEqual(self.ui.radius, radius)
                self.assertTrue(self.ui.message)
            self.ui.action("right")
            self.assertEqual(self.ui.side, 0)
            self.ui.interact(2, 2, 0)
            self.ui.interact(2, 2, 1)
            self.ui.interact(2, 2)
            self.assert_tensors_equal(before)
            self.assert_rng_equal(rng)

    def test_reset_culture_change_exit_and_duel_from_every_phase(self):
        for phase in PHASES:
            for key in (pygame.K_r, pygame.K_6, pygame.K_g, pygame.K_d):
                with self.subTest(phase=phase, key=key):
                    self.reach(phase)
                    self.key(key, pygame.KMOD_SHIFT if key == pygame.K_6 else 0)
                    self.assertEqual(self.world.steps, 0)
                    if key in (pygame.K_r, pygame.K_6):
                        self.assert_seed()
                        self.assertTrue(self.ui.paused)
                        if key == pygame.K_6:
                            self.assertEqual(self.world.left_index, 5)
                            self.assertEqual(self.ui.side, 0)
                    else:
                        self.assertIsNone(self.world.lesson)
                        self.assertEqual((self.world.mode, self.world.size), ("hard", 16))
                        self.assertEqual(bool(self.world.duel), key == pygame.K_d)
                    self.assertEqual(self.ui.effects, [])
                    self.ui.draw(dt=0)
                    self.assertEqual(self.ui.lesson_snapshot, self.world.lesson.snapshot() if self.world.lesson else None)

    def test_failed_ui_selection_preserves_every_phase_even_after_rng_consumption(self):
        def failed(*args, **kwargs):
            torch.rand(17)
            raise ValueError("synthetic unavailable checkpoint")

        for phase in PHASES:
            with self.subTest(phase=phase):
                self.reach(phase)
                before, rng, state = self.tensors(), self.random_states(), self.world.lesson.snapshot()
                paused = self.ui.paused
                with patch("arena.ensure_model", side_effect=failed):
                    self.ui.action(("target", 8))
                self.assertEqual(self.world.lesson.snapshot(), state)
                self.assertEqual(self.ui.lesson_snapshot, state)
                self.assertEqual(self.ui.paused, paused)
                self.assert_tensors_equal(before)
                self.assert_rng_equal(rng)

    def test_recovery_budget_uses_six_ticks_per_second_times_speed(self):
        for speed in (1, 2, 4, 8):
            with self.subTest(speed=speed):
                self.reach("injured")
                self.ui.speed = speed
                self.key(pygame.K_SPACE)
                self.assertEqual(sum(self.ui.step_budget(1 / 12) for _ in range(12)), 6 * speed)
                self.assertEqual(self.world.steps, SHORT.grow_ticks)  # Budget is not a simulation step.

    def test_speed_changes_refresh_the_visible_observation_notice(self):
        self.reach("injured")
        self.key(pygame.K_SPACE)
        self.assertIn("6 simulation ticks", self.ui.message)
        state, rng = self.world.lesson.snapshot(), self.random_states()
        self.key(pygame.K_TAB)
        self.assertEqual(self.ui.speed, 2)
        self.assertIn("12 simulation ticks", self.ui.message)
        self.assertEqual(self.world.lesson.snapshot(), state)
        self.assert_rng_equal(rng)
        with patch.object(self.ui, "text", wraps=self.ui.text) as text:
            self.ui.draw(dt=0)
        self.assertTrue(any("12 simulation ticks" in call.args[0] for call in text.call_args_list))

    def test_budget_clamps_long_frames_and_never_accumulates_pause_catchup(self):
        self.reach("injured")
        self.ui.speed = 1
        self.key(pygame.K_SPACE)
        self.assertEqual(self.ui.step_budget(60), 1)
        self.assertEqual(self.ui.step_budget(1 / 12), 1)
        self.key(pygame.K_SPACE)
        self.assertTrue(self.ui.paused)
        for elapsed in (0, .25, 1, 3600):
            self.assertEqual(self.ui.step_budget(elapsed), 0)
        self.key(pygame.K_SPACE)
        self.assertEqual(self.ui.step_budget(0), 0)
        self.assertEqual(self.ui.step_budget(1 / 12), 0)
        self.assertEqual(self.ui.step_budget(1 / 12), 1)

    def test_growth_uses_normal_pacing_and_action_gates_have_zero_budget(self):
        for phase in PHASES:
            self.reach(phase)
            self.ui.speed = 8
            if phase == "grow":
                self.ui.paused = False
                self.assertEqual(self.ui.step_budget(.001), 8)
            elif phase not in ("grow", "recover"):
                self.assertEqual(self.ui.step_budget(100), 0)

    def test_actual_run_loop_uses_slow_budget_without_changing_draw_fps(self):
        self.reach("injured")
        self.key(pygame.K_SPACE)
        self.world.args.ui_frames = 4
        clock = Mock()
        clock.get_time.return_value = 125
        clock.get_fps.return_value = 30
        self.ui.clock = clock
        with patch("pygame.event.get", return_value=[]), \
                patch.object(self.ui, "draw", wraps=self.ui.draw) as draw:
            self.ui.run()
        self.assertEqual(self.world.steps, SHORT.grow_ticks + 3)
        self.assertEqual(self.world.lesson.phase, "recover")
        self.assertEqual(draw.call_count, 5)  # Initial screen plus all four frames.
        self.assertEqual(clock.tick.call_count, 3)
        for call in clock.tick.call_args_list:
            self.assertEqual(call.args, (self.world.args.fps,))

    def test_draw_fx_colors_resize_and_paused_time_preserve_model_and_rng(self):
        for phase in ("seed", "grow", "damage", "injured", "recover", "complete"):
            self.reach(phase)
            before, rng, state = self.tensors(), self.random_states(), self.world.lesson.snapshot()
            for action in ("colors", "motion", "motion", ("view", "pressure")):
                self.ui.action(action)
                self.ui.draw(dt=.25)
            self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=860, h=740))
            self.assertEqual(self.world.lesson.snapshot(), state)
            self.assertEqual(self.ui.lesson_snapshot, state)
            self.assert_tensors_equal(before)
            self.assert_rng_equal(rng)

    def test_seed_and_crater_cues_remain_visible_with_reduced_motion(self):
        for phase in ("seed", "damage"):
            self.reach(phase)
            self.ui.reduced_motion = True
            self.ui.effects.clear()
            with patch("pygame.mouse.get_pos", return_value=(0, 0)):
                self.ui.draw(dt=.25)
                first = pygame.surfarray.array3d(self.ui.window.subsurface(self.ui.board)).copy()
                self.ui.draw(dt=.25)
                second = pygame.surfarray.array3d(self.ui.window.subsurface(self.ui.board)).copy()
            np.testing.assert_array_equal(first, second)
            raw = pygame.transform.scale(pygame.surfarray.make_surface(self.world.rgb().swapaxes(0, 1)),
                                         self.ui.board.size)
            raw_pixels = pygame.surfarray.array3d(raw)
            # Exclude inset header/footer; only a persistent guided board cue
            # can distinguish this region from the unadorned model pixels.
            margin = self.ui.board.width // 4
            self.assertTrue(np.any(first[margin:-margin, margin:-margin] !=
                                   raw_pixels[margin:-margin, margin:-margin]))

    def test_default_minimum_and_large_layouts_contain_every_control_and_solo_panel(self):
        for phase in ("seed", "damage", "injured", "complete", "timeout", "invalid"):
            self.reach(phase)
            for width, height in ((1100, 902), (200, 200), (1800, 740), (860, 1200)):
                with self.subTest(phase=phase, width=width, height=height):
                    self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
                    window = self.ui.window.get_rect()
                    self.assertTrue(window.contains(self.ui.board))
                    self.assertTrue(window.contains(self.ui.score_rect))
                    self.assertEqual(self.ui.rail_x - self.ui.board.right, 16)
                    for rect, action in self.ui.buttons:
                        self.assertTrue(window.contains(rect), action)
                    self.assertIn("lesson", [action for _, action in self.ui.buttons])
                    self.assertEqual(self.ui.result_buttons, [])
                    self.assertEqual(self.ui.lesson_snapshot["metric"], "living cells")


class LessonCLIIntegrationTests(unittest.TestCase):
    def test_lesson_is_opt_in_and_conflicting_modes_are_rejected(self):
        self.assertFalse(arena.parse_args([]).lesson)
        self.assertTrue(arena.parse_args(["--lesson"]).lesson)
        self.assertTrue(arena.parse_args(["--lesson", "--ui-frames", "1"]).lesson)
        for args in (["--lesson", "--duel"], ["--lesson", "--headless-frames", "1"],
                     ["--lesson", "--headless-frames", "-1"],
                     ["--lesson", "--grid-size", "7"], ["--lesson", "--steps-per-frame", "3"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arena.parse_args(args)


@unittest.skipUnless(os.environ.get("PETRI_TEST_CHECKPOINTS") == "1",
                     "set PETRI_TEST_CHECKPOINTS=1 for bounded shipped-culture lesson smokes")
class ShippedLessonSmokeTests(unittest.TestCase):
    def test_each_ready_culture_has_a_safe_cut_and_a_bounded_finite_observation(self):
        threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
        rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        try:
            targets = clash.list_targets()
            ready = [i for i, target in enumerate(targets) if clash.target_status(target)["status"] == "ready"]
            self.assertTrue(ready)
            for index in ready:
                with self.subTest(culture=targets[index].stem):
                    args = arena.parse_args(["--device", "cpu", "--cpu-threads", "1", "--lesson",
                                             "--seed", "0", "--left", str(index + 1),
                                             "--right", "2" if index == 0 else "1"])
                    world = arena.Arena(args, targets)
                    world.plant_lesson()
                    with patch.dict(world.right, {"model": Mock(side_effect=AssertionError("solo opponent"))}):
                        world.step(1_000)
                        self.assertEqual(world.steps, 160)
                        self.assertEqual(world.lesson.phase, "damage")
                        plan = world.lesson.snapshot()["plan"]
                        self.assertLess(plan["removed"], plan["baseline"])
                        world.cut_lesson()
                        self.assertEqual(world.stats()["left_alive"], plan["remaining"])
                        self.assertEqual(world.steps, 160)
                        world.watch_lesson()
                        world.step(1_000)
                    self.assertIn(world.lesson.phase, ("complete", "timeout"))
                    self.assertLessEqual(world.steps, 320)
                    self.assertTrue(world.report(1)["lesson"]["valid"])
                    self.assertTrue(world.stats()["finite"])
                    self.assertEqual(world.stats()["right_alive"], 0)
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic)
            torch.set_rng_state(rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)


if __name__ == "__main__":
    unittest.main()
