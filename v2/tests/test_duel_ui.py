"""CPU/SDL integration contracts for seeded duels and their result controls.

Synthetic NCAs have nonzero updates, so replay tests exercise real stochastic
state changes rather than a stationary zero-initialized network. These tests
do not measure trained-model quality or native display performance. The single
shipped-checkpoint smoke test is opt-in with PETRI_TEST_CHECKPOINTS=1.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import numpy as np
import pygame
import torch

import arena
import clash
from nca import NCA


def synthetic_bundle(target, device, *args, **kwargs):
    # Model construction is deterministic independently of simulation seed,
    # model selection order, and the process's existing random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sum(map(ord, target.stem)))
        model = NCA(channels=8, hidden_size=16).to(device).eval()
        with torch.no_grad():
            model.fc1.weight.normal_(std=.015)
            model.fc1.bias[3] = .06
    return {"model": model, "channels": 8, "grid_size": 16,
            "channels_last": False, "name": target.stem,
            "health": "ready", "seed_dir": "seed_000", "score": .01,
            "kind": "synthetic-untrained", "source": "duel-ui-test"}


class DuelUIIntegrationTests(unittest.TestCase):
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
        args = arena.parse_args([
            "--device", "cpu", "--cpu-threads", "1", "--grid-size", "16",
            "--duel", "--round-ticks", "19", "--warmup-ticks", "3",
            "--seed", "27", *extra])
        self.world = arena.Arena(args, clash.list_targets())
        self.ui = arena.ArenaUI(self.world)
        self.ui.draw(dt=0)

    def key(self, key, mod=0):
        self.ui.event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))

    def click(self, pos, button=1):
        self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))

    def control(self, action, *, result=False):
        controls = self.ui.result_buttons if result else self.ui.buttons
        return next(rect for rect, value in controls if value == action)

    def tensors(self):
        return tuple(value.clone() for value in (
            self.world.a, self.world.b, self.world.owner, self.world.control))

    def assert_tensors_equal(self, expected):
        for actual, previous in zip(self.tensors(), expected):
            self.assertTrue(torch.equal(actual, previous))

    def finish(self):
        self.world.step(self.world.duel_config.total_ticks + 8)
        self.ui.draw(dt=.01)
        self.assertTrue(self.world.duel.finished)
        self.assertTrue(self.ui.paused)

    def assert_fresh_round(self, seed=27):
        self.assertEqual(self.world.args.seed, seed)
        self.assertEqual(self.world.steps, 0)
        self.assertFalse(self.world.duel.finished)
        self.assertFalse(self.ui.paused)
        self.assertTrue(self.ui.result_visible)
        self.assertEqual(self.ui.duel_snapshot, self.world.duel.snapshot())
        self.assertEqual(self.ui.duel_snapshot["scores"], {"left": 0, "right": 0})
        self.assertEqual(self.ui.duel_snapshot["scored_ticks"], 0)
        self.assertIsNone(self.ui.duel_snapshot["winner"])
        self.assertEqual(self.ui.effects, [])
        self.assertEqual(self.ui.notices, [])
        self.assertEqual(self.ui.summary["span"], 0)
        self.assertIsNone(self.ui.blend.rendered)

    def test_start_button_and_d_toggle_reset_into_and_out_of_duel(self):
        initial = self.tensors()
        for start in ("keyboard", "button"):
            with self.subTest(start=start):
                self.key(pygame.K_d)
                self.assertIsNone(self.world.duel)
                self.assertFalse(self.world.duel_enabled)
                self.world.clear()
                self.world.plant(4, 4, 0)
                self.world.step(2)
                self.ui.draw(dt=0)
                if start == "keyboard":
                    self.key(pygame.K_d)
                else:
                    self.click(self.control("duel").center)
                self.assert_fresh_round()
                self.assert_tensors_equal(initial)
                self.assertTrue(self.world.duel_enabled)

    def test_paused_warmup_draws_and_single_steps_use_simulation_ticks(self):
        self.key(pygame.K_SPACE)
        before = self.tensors()
        random_state = torch.get_rng_state().clone()
        for _ in range(5):
            self.ui.draw(dt=.25)
        self.assertEqual(self.world.steps, 0)
        self.assertEqual(self.ui.duel_snapshot["phase"], "warmup")
        self.assertIn("WARMUP", self.ui.duel_headline())
        self.assert_tensors_equal(before)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))
        for tick in range(1, 4):
            self.key(pygame.K_n)
            self.assertTrue(self.ui.paused)
            self.assertEqual(self.world.steps, tick)
            self.assertEqual(self.ui.duel_snapshot["scored_ticks"], 0)
            self.assertEqual(self.ui.duel_snapshot["scores"], {"left": 0, "right": 0})
        self.click(self.control("step").center)
        self.assertTrue(self.ui.paused)
        self.assertEqual(self.world.steps, 4)
        self.assertEqual(self.ui.duel_snapshot["scored_ticks"], 1)
        stats = self.world.stats()
        self.assertEqual(self.ui.duel_snapshot["scores"],
                         {"left": stats["left_territory"], "right": stats["right_territory"]})
        snapshot = self.world.duel.snapshot()
        for _ in range(3):
            self.ui.draw(dt=.25)
        self.assertEqual(self.world.duel.snapshot(), snapshot)

    def test_every_speed_and_extra_draws_replay_identical_scores_and_tensors(self):
        for mode in ("hard", "soft"):
            with self.subTest(mode=mode):
                self.new_world("--mode", mode)
                self.assert_speed_replays()

    def assert_speed_replays(self):
        initial = self.tensors()
        totals = {"left": 0, "right": 0}
        metric = "territory" if self.world.mode == "hard" else "alive"
        for tick in range(1, self.world.duel_config.total_ticks + 1):
            self.world.step()
            if tick > self.world.duel_config.warmup_ticks:
                stats = self.world.stats()
                totals["left"] += stats[f"left_{metric}"]
                totals["right"] += stats[f"right_{metric}"]
        expected = self.tensors()
        expected_result = self.world.duel.snapshot()
        expected_rng = torch.get_rng_state().clone()
        self.assertEqual(expected_result["scores"], totals)
        self.assertFalse(torch.equal(initial[0], expected[0]))
        self.assertFalse(torch.equal(initial[1], expected[1]))
        for speed in (1, 2, 4, 8):
            with self.subTest(speed=speed):
                self.ui.action("reset")
                while self.ui.speed != speed:
                    self.key(pygame.K_TAB)
                self.assertEqual(self.world.steps, 0)
                while not self.world.duel.finished:
                    remaining = self.world.duel_config.total_ticks - self.world.steps
                    self.assertEqual(self.world.step(self.ui.speed), min(speed, remaining))
                    self.ui.action("colors")
                    self.ui.action("motion")
                    self.ui.draw(dt=.01)
                    self.ui.draw(dt=.2)
                self.assert_tensors_equal(expected)
                self.assertEqual(self.world.duel.snapshot(), expected_result)
                self.assertEqual(self.ui.duel_snapshot, expected_result)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertTrue(self.ui.paused)

    def test_view_draws_preserve_duel_state_and_all_random_generators(self):
        self.world.step(7)
        tensors, result = self.tensors(), self.world.duel.snapshot()
        torch_rng = torch.get_rng_state().clone()
        numpy_rng, python_rng = np.random.get_state(), random.getstate()
        for view in ("organisms", "territory", "pressure"):
            self.ui.action(("view", view))
            for _ in range(3):
                self.ui.draw(dt=.25)
        self.assert_tensors_equal(tensors)
        self.assertEqual(self.world.duel.snapshot(), result)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
        self.assertEqual(numpy_rng[2:], current_numpy[2:])
        self.assertEqual(python_rng, random.getstate())

    def test_finished_round_is_paused_and_all_advances_and_edits_stay_frozen(self):
        self.finish()
        tensors, result = self.tensors(), self.world.duel.snapshot()
        random_state = torch.get_rng_state().clone()
        for key in (pygame.K_SPACE, pygame.K_n, pygame.K_c, pygame.K_SPACE):
            self.key(key)
            self.assertTrue(self.ui.paused)
        self.click(self.ui.board.topleft, button=3)
        self.ui.interact(5, 5)
        self.ui.interact(5, 5, 0)
        self.assertEqual(self.world.step(1000), 0)
        self.ui.draw(dt=.25)
        self.assert_tensors_equal(tensors)
        self.assertEqual(self.world.duel.snapshot(), result)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))

    def test_nonfinite_hard_model_ends_invalid_instead_of_awarding_a_winner(self):
        with torch.no_grad():
            self.world.left["model"].fc1.bias[3] = float("nan")
        self.world.step(19)
        self.ui.draw(dt=0)
        result = self.world.duel.snapshot()
        self.assertTrue(result["finished"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["phase"], "invalid")
        self.assertIsNone(result["winner"])
        self.assertIsNone(result["final"])
        self.assertEqual(result["scored_ticks"], 0)
        self.assertEqual(result["scores"], {"left": 0, "right": 0})
        self.assertLess(self.world.steps, self.world.duel_config.total_ticks)
        self.assertTrue(self.ui.paused)
        self.assertEqual(self.ui.duel_headline(), "ROUND INVALID")
        self.assertIsNotNone(self.ui.result_rect)
        self.assertEqual(self.world.step(1000), 0)
        self.assertEqual(self.world.duel.snapshot(), result)
        with patch.object(self.ui, "text", wraps=self.ui.text) as text:
            self.ui.draw(dt=0)
        labels = [call.args[0] for call in text.call_args_list]
        self.assertIn("Non-finite state. No winner awarded.", labels)
        self.assertNotIn("AVERAGE HELD CELLS", labels)

    def test_duel_api_rejects_edits_without_mutating_round(self):
        self.world.step(5)
        tensors, result = self.tensors(), self.world.duel.snapshot()
        for operation in (self.world.clear,
                          lambda: self.world.plant(5, 5, 0),
                          lambda: self.world.damage(5, 5, 2)):
            with self.assertRaisesRegex(ValueError, "locked"):
                operation()
            self.assert_tensors_equal(tensors)
            self.assertEqual(self.world.duel.snapshot(), result)

    def test_result_buttons_are_hit_before_underlying_board(self):
        for action in ("reset", "next_seed", "duel"):
            with self.subTest(action=action):
                self.new_world()
                self.finish()
                rect = self.control(action, result=True)
                self.assertTrue(self.ui.board.contains(rect))
                with patch.object(self.ui, "interact", wraps=self.ui.interact) as interact, \
                        patch.object(self.world, "damage", wraps=self.world.damage) as damage:
                    self.click(rect.center)
                interact.assert_not_called()
                damage.assert_not_called()
                self.assertEqual(self.world.steps, 0)
                self.assertFalse(self.ui.paused)
                if action == "duel":
                    self.assertIsNone(self.world.duel)
                else:
                    self.assert_fresh_round(seed=28 if action == "next_seed" else 27)

    def test_r_enter_and_result_click_rematch_replay_exactly(self):
        initial = self.tensors()
        self.finish()
        expected, result = self.tensors(), self.world.duel.snapshot()
        expected_rng = torch.get_rng_state().clone()
        for trigger in (pygame.K_r, pygame.K_RETURN, "click"):
            with self.subTest(trigger=trigger):
                if trigger == "click":
                    self.click(self.control("reset", result=True).center)
                else:
                    self.key(trigger)
                self.assert_fresh_round()
                self.assert_tensors_equal(initial)
                self.ui.draw(dt=0)
                self.assertIsNone(self.ui.result_rect)
                self.assertEqual(self.ui.result_buttons, [])
                self.finish()
                self.assert_tensors_equal(expected)
                self.assertEqual(self.world.duel.snapshot(), result)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_next_seed_stale_doubleclick_cannot_increment_or_restart_new_round(self):
        self.finish()
        position = self.control("next_seed", result=True).center
        self.click(position)
        self.assert_fresh_round(seed=28)
        active_round = self.world.duel
        # Keep the old drawn hit rectangles, as with two events in one batch.
        self.world.step(2)
        expected, result = self.tensors(), active_round.snapshot()
        self.click(position)
        self.ui.action("next_seed")
        self.assertIs(self.world.duel, active_round)
        self.assertEqual(self.world.args.seed, 28)
        self.assertEqual(self.world.steps, 2)
        self.assert_tensors_equal(expected)
        self.assertEqual(self.world.duel.snapshot(), result)
        self.assertFalse(self.ui.paused)

    def test_hide_and_reopen_result_preserve_frozen_state_scores_and_rng(self):
        self.finish()
        tensors, result = self.tensors(), self.world.duel.snapshot()
        torch_rng = torch.get_rng_state().clone()
        numpy_rng, python_rng = np.random.get_state(), random.getstate()
        old_next_seed = self.control("next_seed", result=True).center
        with patch.object(self.ui, "interact", wraps=self.ui.interact) as interact:
            self.click(self.control("result", result=True).center)
        interact.assert_not_called()
        self.assertFalse(self.ui.result_visible)
        # Old result rectangles must become inert immediately, before a draw.
        self.click(old_next_seed)
        self.assertEqual(self.world.args.seed, 27)
        self.ui.draw(dt=.25)
        self.assertIsNone(self.ui.result_rect)
        self.assertEqual(self.ui.result_buttons, [])
        for view in ("territory", "pressure", "organisms"):
            self.ui.action(("view", view))
            self.ui.draw(dt=.25)
        self.key(pygame.K_SPACE)
        self.assertTrue(self.ui.paused)
        self.click(self.control("result").center)
        self.assertTrue(self.ui.result_visible)
        self.ui.draw(dt=.25)
        self.assertIsNotNone(self.ui.result_rect)
        self.assertEqual(len(self.ui.result_buttons), 4)
        self.assert_tensors_equal(tensors)
        self.assertEqual(self.world.duel.snapshot(), result)
        self.assertEqual(self.ui.duel_snapshot, result)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
        self.assertEqual(numpy_rng[2:], current_numpy[2:])
        self.assertEqual(python_rng, random.getstate())

    def test_r_and_enter_rematch_when_result_is_hidden(self):
        initial = self.tensors()
        for key in (pygame.K_r, pygame.K_RETURN):
            with self.subTest(key=key):
                self.finish()
                self.ui.action("result")
                self.ui.draw(dt=0)
                self.assertFalse(self.ui.result_visible)
                self.assertIsNone(self.ui.result_rect)
                self.key(key)
                self.assert_fresh_round()
                self.assert_tensors_equal(initial)

    def test_next_seed_wraps_to_valid_seed_and_still_only_runs_once(self):
        self.world.args.seed = 2 ** 32 - 1
        self.ui.action("reset")
        self.finish()
        position = self.control("next_seed", result=True).center
        self.click(position)
        self.assert_fresh_round(seed=0)
        self.click(position)
        self.assertEqual(self.world.args.seed, 0)
        self.assertEqual(self.world.steps, 0)

    def test_back_to_lab_restores_clear_plant_and_damage(self):
        self.finish()
        self.click(self.control("duel", result=True).center)
        self.assertIsNone(self.world.duel)
        self.assertIsNone(self.ui.duel_snapshot)
        self.assertFalse(self.world.duel_enabled)
        self.ui.action("clear")
        self.ui.interact(5, 5, 0)
        self.assertEqual(self.world.stats()["left_alive"], 1)
        self.ui.interact(9, 9, 1)
        self.assertEqual(self.world.stats()["right_alive"], 1)
        self.ui.interact(5, 5)
        self.assertEqual(self.world.stats()["left_alive"], 0)
        self.assertEqual(self.world.stats()["right_alive"], 1)
        self.ui.draw(dt=0)
        self.assertIsNone(self.ui.result_rect)
        self.assertEqual(self.ui.result_buttons, [])
        self.assertEqual(self.ui.stats_cache, self.world.stats())

    def test_back_to_lab_doubleclick_does_not_damage_the_fresh_sandbox(self):
        self.finish()
        position = self.control("duel", result=True).center
        with patch.object(self.ui, "interact", wraps=self.ui.interact) as interact, \
                patch.object(self.world, "damage", wraps=self.world.damage) as damage:
            self.click(position)
            self.assertIsNone(self.world.duel)
            fresh_lab = self.tensors()
            # Both clicks can be delivered before the old result card redraws.
            self.click(position)
        interact.assert_not_called()
        damage.assert_not_called()
        self.assert_tensors_equal(fresh_lab)
        self.assertEqual(self.world.steps, 0)
        self.ui.draw(dt=0)
        self.ui.interact(5, 5, 0)
        self.assertEqual(self.world.a[0, 3, 5, 5].item(), 1)

    def test_failed_culture_selection_preserves_running_or_paused_duel(self):
        self.world.step(7)
        self.ui.refresh()
        tensors, result = self.tensors(), self.world.duel.snapshot()
        current_round = self.world.duel
        left, right = self.world.left, self.world.right
        for paused in (False, True):
            for side in (0, 1):
                with self.subTest(paused=paused, side=side):
                    self.ui.paused, self.ui.side = paused, side
                    with patch.object(self.world, "load", side_effect=ValueError("unusable checkpoint")):
                        self.ui.action(("target", 2))
                    self.assertIs(self.world.duel, current_round)
                    self.assertIs(self.world.left, left)
                    self.assertIs(self.world.right, right)
                    self.assertEqual(self.ui.paused, paused)
                    self.assert_tensors_equal(tensors)
                    self.assertEqual(self.world.duel.snapshot(), result)
                    self.assertEqual(self.ui.duel_snapshot, result)
                    self.assertIn("checkpoint", self.ui.message)

    def test_failure_after_model_construction_preserves_rng_and_future_duel(self):
        self.world.step(7)
        self.world.step(100)
        expected, result = self.tensors(), self.world.duel.snapshot()
        self.ui.action("reset")
        self.world.step(7)
        current_round = self.world.duel
        random_state = torch.get_rng_state().clone()

        def fail_after_construction(*args, **kwargs):
            # This matches loading a malformed state_dict: its NCA constructor
            # has already consumed randomness before the load error occurs.
            NCA(channels=8, hidden_size=16)
            raise ValueError("malformed checkpoint weights")

        with patch("arena.ensure_model", side_effect=fail_after_construction):
            self.ui.action(("target", 2))
        self.assertIs(self.world.duel, current_round)
        self.assertEqual(self.world.steps, 7)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))
        self.world.step(100)
        self.assert_tensors_equal(expected)
        self.assertEqual(self.world.duel.snapshot(), result)

    def test_culture_and_mode_changes_start_clean_rounds_without_changing_seed(self):
        for action in (("target", 2), "mode"):
            with self.subTest(action=action):
                self.finish()
                self.ui.action(action)
                self.assertEqual(self.world.steps, 0)
                self.assertEqual(self.world.args.seed, 27)
                self.assertTrue(self.world.duel_enabled)
                self.assertFalse(self.world.duel.finished)
                self.assertFalse(self.ui.paused)
                self.assertEqual(self.world.duel.snapshot()["scores"], {"left": 0, "right": 0})
                self.assertEqual(self.world.duel.snapshot()["scored_ticks"], 0)
                self.assertEqual(self.ui.effects, [])
                self.assertEqual(self.ui.summary["span"], 0)
        self.assertEqual(self.world.mode, "soft")
        self.assertEqual(self.world.duel.snapshot()["metric"], "living cells")
        self.assertEqual(self.ui.view, "organisms")

    def test_soft_duel_scores_overlapping_life_and_labels_the_metric(self):
        self.new_world("--mode", "soft", "--left-pos", "5,5", "--right-pos", "5,5")
        # Hold one overlapping live cell per side to make the scoring oracle
        # independent of either the ownership map or learned dynamics.
        with torch.no_grad():
            for bundle in (self.world.left, self.world.right):
                bundle["model"].fc1.weight.zero_()
                bundle["model"].fc1.bias.zero_()
        self.finish()
        snapshot = self.world.duel.snapshot()
        self.assertEqual(self.world.stats()["contested"], 1)
        self.assertIsNone(self.world.stats()["left_territory"])
        self.assertIsNone(self.world.stats()["right_territory"])
        self.assertEqual(snapshot["metric"], "living cells")
        self.assertEqual(snapshot["scores"], {"left": 16, "right": 16})
        self.assertEqual(snapshot["averages"], {"left": 1.0, "right": 1.0})
        self.assertEqual(snapshot["winner"], "draw")
        self.assertEqual(self.ui.summary["metric"], "living cells")
        self.assertEqual(self.world.report(1)["placement"], "custom")
        with patch.object(self.ui, "text", wraps=self.ui.text) as text:
            self.ui.draw(dt=0)
        labels = [call.args[0] for call in text.call_args_list]
        self.assertIn("MEAN LIVING CELLS", labels)
        self.assertIn("AVERAGE LIVING CELLS", labels)
        self.assertIn("ROUND DRAW", labels)
        self.assertNotIn("AVERAGE HELD CELLS", labels)

    def test_default_duel_positions_are_mirrored_and_explicit_positions_survive_reset(self):
        for size in (8, 15, 16):
            with self.subTest(size=size):
                self.new_world("--grid-size", str(size))
                left = torch.nonzero(self.world.a[0, 3] > .1).tolist()
                right = torch.nonzero(self.world.b[0, 3] > .1).tolist()
                self.assertEqual(left, [[(size - 1) // 2, size // 3]])
                self.assertEqual(right, [[(size - 1) // 2, size - 1 - size // 3]])
                self.assertEqual(self.world.report(1)["placement"], "mirrored")
        self.new_world("--left-pos", "3,4", "--right-pos", "11,10")
        initial = self.tensors()
        self.world.step(5)
        self.ui.action("reset")
        self.assert_tensors_equal(initial)
        self.assertEqual(torch.nonzero(self.world.a[0, 3] > .1).tolist(), [[4, 3]])
        self.assertEqual(torch.nonzero(self.world.b[0, 3] > .1).tolist(), [[10, 11]])
        self.assertEqual(self.world.report(1)["placement"], "custom")

    def test_resize_keeps_results_and_controls_contained_with_fresh_hit_geometry(self):
        self.finish()
        for width, height in ((200, 200), (860, 740), (1100, 902), (1800, 740), (860, 1200)):
            with self.subTest(size=(width, height)):
                self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
                window = self.ui.window.get_rect()
                self.assertTrue(window.contains(self.ui.board))
                self.assertTrue(window.contains(self.ui.result_rect))
                self.assertTrue(self.ui.board.contains(self.ui.result_rect))
                self.assertEqual([action for _, action in self.ui.result_buttons],
                                 ["result", "reset", "next_seed", "duel"])
                for index, (rect, _) in enumerate(self.ui.result_buttons):
                    self.assertTrue(self.ui.result_rect.contains(rect))
                    self.assertTrue(window.contains(rect))
                    for other, _ in self.ui.result_buttons[index + 1:]:
                        self.assertFalse(rect.colliderect(other))
                for rect, _ in self.ui.buttons:
                    self.assertTrue(window.contains(rect))
        with patch.object(self.ui, "interact", wraps=self.ui.interact) as interact:
            self.click(self.control("reset", result=True).center)
        interact.assert_not_called()
        self.assert_fresh_round()

    def test_run_loop_finishes_at_horizon_and_stays_paused_for_remaining_frames(self):
        self.world.args.ui_frames = 8
        self.ui.speed = 8
        # Exercise the actual loop without sleeping or native window events.
        with patch("pygame.event.get", return_value=[]), \
                patch.object(self.ui, "clock") as clock:
            clock.get_fps.return_value = 30
            self.ui.run()
        self.assertEqual(self.world.steps, 19)
        self.assertTrue(self.world.duel.finished)
        self.assertTrue(self.ui.paused)
        self.assertEqual(self.ui.duel_snapshot["scored_ticks"], 16)


class DuelCLIIntegrationTests(unittest.TestCase):
    def test_cli_rejects_invalid_round_and_warmup_bounds(self):
        invalid = (["--round-ticks", "0"], ["--round-ticks", "-1"],
                   ["--round-ticks", "1.5"], ["--warmup-ticks", "-1"],
                   ["--round-ticks", "7", "--warmup-ticks", "7"],
                   ["--round-ticks", "7", "--warmup-ticks", "8"],
                   ["--round-ticks", "1"])
        for args in invalid:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                arena.parse_args(["--duel", *args])
            self.assertEqual(error.exception.code, 2)

    def test_cli_accepts_one_scoring_tick_and_defaults(self):
        for total, warmup in ((1, 0), (7, 0), (7, 6)):
            with self.subTest(total=total, warmup=warmup):
                args = arena.parse_args(["--duel", "--round-ticks", str(total),
                                         "--warmup-ticks", str(warmup)])
                self.assertTrue(args.duel)
                self.assertEqual((args.round_ticks, args.warmup_ticks), (total, warmup))
        args = arena.parse_args([])
        self.assertFalse(args.duel)
        self.assertEqual((args.round_ticks, args.warmup_ticks), (600, 60))

    def test_headless_reports_clip_and_reject_invalid_results_without_importing_sdl(self):
        # A fresh interpreter makes an accidental pygame import observable,
        # even when another integration test already initialized SDL here.
        with tempfile.TemporaryDirectory() as folder:
            script = textwrap.dedent("""
                import importlib.abc
                from pathlib import Path
                import sys

                class NoPygame(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "pygame" or fullname.startswith("pygame."):
                            raise AssertionError("SDL dependency imported by headless duel")

                sys.meta_path.insert(0, NoPygame())
                sys.path.insert(0, sys.argv[1])
                import arena
                from nca import NCA

                def bundle(target, device, *args, **kwargs):
                    return {"model": NCA(channels=8, hidden_size=16).to(device).eval(),
                            "channels": 8, "grid_size": 16, "channels_last": False,
                            "name": target.stem, "health": "ready", "seed_dir": "seed_000",
                            "score": .01, "source": "headless-duel-test"}

                def no_ui(*args, **kwargs):
                    raise AssertionError("UI constructed by headless duel")

                arena.ensure_model = bundle
                arena.ArenaUI = no_ui
                output = Path(sys.argv[2])

                def run(name, mode, ticks):
                    arena.main(["--device", "cpu", "--duel", "--mode", mode,
                                "--round-ticks", "9", "--warmup-ticks", "2",
                                "--headless-frames", str(ticks), "--fps", "1",
                                "--report", str(output / (name + ".json")),
                                "--snapshot", str(output / (name + ".png"))])

                if len(sys.argv) > 3 and sys.argv[3] == "invalid":
                    import torch
                    def nonfinite_bundle(*args, **kwargs):
                        value = bundle(*args, **kwargs)
                        with torch.no_grad():
                            value["model"].fc1.bias[3] = float("nan")
                        return value
                    arena.ensure_model = nonfinite_bundle
                    run("invalid", "hard", 1000)
                else:
                    for name, mode, ticks in (("hard", "hard", 1000),
                                              ("soft", "soft", 1000),
                                              ("partial", "hard", 4)):
                        run(name, mode, ticks)
                assert "pygame" not in sys.modules
            """)
            result = subprocess.run(
                [sys.executable, "-c", script, str(V2_ROOT), folder],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "SDL_VIDEODRIVER": "intentionally-unavailable"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name, metric in (("hard", "held cells"), ("soft", "living cells")):
                report = json.loads((Path(folder) / f"{name}.json").read_text())
                self.assertEqual(report["step"], 9)
                self.assertTrue(report["finite"])
                self.assertEqual(report["placement"], "mirrored")
                self.assertEqual(report["duel"]["phase"], "finished")
                self.assertEqual(report["duel"]["scores"], {"left": 7, "right": 7})
                self.assertEqual(report["duel"]["scored_ticks"], 7)
                self.assertEqual(report["duel"]["metric"], metric)
                self.assertEqual(report["duel"]["winner"], "draw")
                self.assertEqual(report["duel"]["remaining_ticks"], 0)
                self.assertTrue((Path(folder) / f"{name}.png").is_file())
            partial = json.loads((Path(folder) / "partial.json").read_text())
            self.assertEqual(partial["step"], 4)
            self.assertEqual(partial["duel"]["scored_ticks"], 2)
            self.assertFalse(partial["duel"]["finished"])
            self.assertIsNone(partial["duel"]["winner"])
            self.assertIsNone(partial["duel"]["final"])
            invalid_process = subprocess.run(
                [sys.executable, "-c", script, str(V2_ROOT), folder, "invalid"],
                capture_output=True, text=True, timeout=60,
                env={**os.environ, "SDL_VIDEODRIVER": "intentionally-unavailable"})
            self.assertNotEqual(invalid_process.returncode, 0)
            invalid_path = Path(folder) / "invalid.json"
            self.assertTrue(invalid_path.is_file(), invalid_process.stdout + invalid_process.stderr)
            invalid = json.loads(invalid_path.read_text())
            self.assertTrue(invalid["duel"]["finished"])
            self.assertFalse(invalid["duel"]["valid"])
            self.assertEqual(invalid["duel"]["phase"], "invalid")
            self.assertIsNone(invalid["duel"]["winner"])
            self.assertLess(invalid["step"], 9)


@unittest.skipUnless(os.environ.get("PETRI_TEST_CHECKPOINTS") == "1",
                     "set PETRI_TEST_CHECKPOINTS=1 for the shipped-model duel smoke test")
class TrainedDuelSmokeTests(unittest.TestCase):
    def tearDown(self):
        pygame.quit()

    def test_real_models_replay_identical_round_at_different_frame_speeds(self):
        args = arena.parse_args(["--device", "cpu", "--cpu-threads", "1", "--duel",
                                 "--seed", "17", "--round-ticks", "49", "--warmup-ticks", "8"])
        world = arena.Arena(args, clash.list_targets())
        self.assertEqual(world.left["health"], "ready")
        self.assertEqual(world.right["health"], "ready")
        ui = arena.ArenaUI(world)
        world.step(49)
        expected = tuple(value.clone() for value in (world.a, world.b, world.owner, world.control))
        result = world.duel.snapshot()
        ui.action("reset")
        ui.speed = 8
        while not world.duel.finished:
            world.step(ui.speed)
            ui.draw(dt=.01)
        for actual, previous in zip((world.a, world.b, world.owner, world.control), expected):
            self.assertTrue(torch.equal(actual, previous))
        self.assertEqual(world.duel.snapshot(), result)
        self.assertEqual(world.steps, 49)
        self.assertEqual(result["scored_ticks"], 41)
        self.assertTrue(result["valid"])
        self.assertTrue(ui.paused)


if __name__ == "__main__":
    unittest.main()
