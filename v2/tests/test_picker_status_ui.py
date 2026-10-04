"""CPU/dummy-SDL regressions for checkpoint-pinned picker presentation.

Discovery, health policy, selection, and export use the shipped metadata. Only
checkpoint tensor loading is replaced with a tiny NCA, so these are lifecycle
and UI integration tests, not trained-model quality or native-desktop checks.
"""

import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

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
from nca import NCA


def tiny_checkpoint(checkpoint, device):
    """Keep actual resolved checkpoint provenance without deserializing weights."""
    checkpoint = Path(checkpoint)
    return {"model": NCA(channels=8, hidden_size=16).to(device).eval(),
            "channels": 8, "grid_size": 16, "channels_last": False,
            "kind": "synthetic-picker-test", "source": str(checkpoint),
            "seed_dir": checkpoint.parents[1].name,
            "score": clash.seed_score(checkpoint.parents[1])}


class PickerStatusUITests(unittest.TestCase):
    def setUp(self):
        self.original_rng = self.random_states()
        self.threads = torch.get_num_threads()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.loader = patch("clash.load_v2_model", side_effect=tiny_checkpoint)
        self.load = self.loader.start()
        self.targets = clash.list_targets()
        self.heart, self.star, self.sun, self.flower, self.umbrella = (
            next(i for i, target in enumerate(self.targets) if target.stem.endswith(name))
            for name in ("heart", "star", "sun", "flower", "umbrella"))

    def tearDown(self):
        pygame.quit()
        self.loader.stop()
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(self.deterministic)
        torch.set_rng_state(self.original_rng[0])
        np.random.set_state(self.original_rng[1])
        random.setstate(self.original_rng[2])

    def new_world(self, *extra):
        args = arena.parse_args(["--device", "cpu", "--cpu-threads", "1",
                                 "--grid-size", "16", "--seed", "27",
                                 "--round-ticks", "7", "--warmup-ticks", "2", *extra])
        self.world = arena.Arena(args, self.targets)
        self.ui = arena.ArenaUI(self.world)
        self.ui.draw(dt=0)

    def pinned_world(self, *extra):
        self.new_world("--left-seed", "0", "--right-seed", "1", *extra)

    def lesson_world(self, *extra):
        self.new_world("--lesson", "--left", str(self.star + 1),
                       "--left-seed", "1", "--right", str(self.umbrella + 1),
                       "--right-seed", "2", *extra)

    @staticmethod
    def random_states():
        return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()

    def assert_rng_equal(self, expected):
        current = self.random_states()
        self.assertTrue(torch.equal(current[0], expected[0]))
        self.assertEqual(current[1][0], expected[1][0])
        np.testing.assert_array_equal(current[1][1], expected[1][1])
        self.assertEqual(current[1][2:], expected[1][2:])
        self.assertEqual(current[2], expected[2])

    def frozen_state(self):
        return {"left": self.world.left, "right": self.world.right,
                "indices": (self.world.left_index, self.world.right_index),
                "pins": (self.world.args.left_seed, self.world.args.right_seed),
                "steps": self.world.steps, "duel": self.world.duel,
                "duel_state": self.world.duel.snapshot() if self.world.duel else None,
                "tensors": [(value, value.clone()) for value in
                            (self.world.a, self.world.b, self.world.owner, self.world.control)],
                "rng": self.random_states()}

    def assert_frozen(self, before):
        self.assertIs(self.world.left, before["left"])
        self.assertIs(self.world.right, before["right"])
        self.assertEqual((self.world.left_index, self.world.right_index), before["indices"])
        self.assertEqual((self.world.args.left_seed, self.world.args.right_seed), before["pins"])
        self.assertEqual(self.world.steps, before["steps"])
        self.assertIs(self.world.duel, before["duel"])
        if self.world.duel:
            self.assertEqual(self.world.duel.snapshot(), before["duel_state"])
        for actual, (original, expected) in zip(
                (self.world.a, self.world.b, self.world.owner, self.world.control), before["tensors"]):
            self.assertIs(actual, original)
            self.assertTrue(torch.equal(actual, expected))
        self.assert_rng_equal(before["rng"])

    def key(self, key, mod=0):
        self.ui.event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))

    def click_target(self, index):
        rect = next(rect for rect, action in self.ui.buttons if action == ("target", index))
        self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=rect.center, button=1))

    def texts(self, draw=None):
        """Record the strings actually visible after the production clipping rule."""
        with patch.object(self.ui, "text", wraps=self.ui.text) as text:
            (draw or (lambda: self.ui.draw(dt=0)))()
        result = []
        for call in text.call_args_list:
            value, x, y = str(call.args[0]), call.args[1], call.args[2]
            size = call.args[3] if len(call.args) > 3 else call.kwargs.get("size", 14)
            width = call.args[5] if len(call.args) > 5 else call.kwargs.get("width")
            if width is not None:
                while value and self.ui.fonts[size].size(value)[0] > width:
                    value = value[:-2] + "…" if len(value) > 1 else ""
            result.append((value, pygame.Rect((x, y), self.ui.fonts[size].size(value))))
        return result

    def assert_identity(self, text, culture, seed, health):
        self.assertIn(culture, text.lower())
        self.assert_checkpoint(text, seed, health)

    def assert_checkpoint(self, text, seed, health):
        self.assertRegex(text.lower(), rf"\bseed[_ ]+0*{seed}(?!\d)")
        self.assertIn(health, text.lower())

    def underboard_text(self):
        return " ".join(text for text, _ in self.texts(self.ui.draw_underboard))

    def test_side_switch_uses_each_pin_for_status_and_thumbnail_alpha(self):
        self.pinned_world()
        for side, expected in (("left", ((self.star, "collapsed", 0, 60),
                                          (self.umbrella, "ready", 0, 255))),
                               ("right", ((self.star, "ready", 1, 255),
                                           (self.umbrella, "missing", None, 60)))):
            self.ui.action(side)
            self.ui.draw(dt=0)
            for index, health, seed, alpha in expected:
                with self.subTest(side=side, culture=self.targets[index].stem):
                    self.assertEqual(self.ui.status[index]["status"], health)
                    self.assertEqual(self.ui.status[index]["seed"], seed)
                    self.assertEqual(self.ui.status[index]["selectable"], health == "ready")
                    self.assertEqual(self.ui.thumbs[index].get_alpha(), alpha)
            policy = next(text for text, _ in self.texts() if "PIN" in text)
            self.assertIn("PIN 000" if side == "left" else "PIN 001", policy)

    def test_default_auto_selection_and_labels_keep_best_healthy_checkpoints(self):
        self.new_world()
        self.assertIsNone(self.world.args.left_seed)
        self.assertIsNone(self.world.args.right_seed)
        for index, seed in ((self.star, 1), (self.sun, 2), (self.umbrella, 0)):
            with self.subTest(culture=self.targets[index].stem):
                self.assertEqual(self.ui.status[index]["status"], "ready")
                self.assertEqual(self.ui.status[index]["seed"], seed)
                self.ui.action(("target", index))
                self.assertEqual(self.world.left["seed_dir"], f"seed_{seed:03d}")
                self.assert_identity(self.ui.message, self.targets[index].stem.split("_", 1)[1], seed, "ready")
        self.assertTrue(any("AUTO" in text for text, _ in self.texts()))

    def test_successful_selection_reports_exact_loaded_checkpoint(self):
        self.pinned_world()
        self.click_target(self.umbrella)
        self.assertEqual(self.world.left["name"], "07_umbrella")
        self.assertEqual(self.world.left["seed_dir"], "seed_000")
        self.assert_identity(self.ui.message, "umbrella", 0, "ready")
        self.assert_checkpoint(self.underboard_text(), 0, "ready")
        self.assertEqual((self.world.args.left_seed, self.world.args.right_seed), (0, 1))
        self.assertIn("/07_umbrella/seed_000/", self.world.left["source"])

    def test_override_keeps_collapsed_health_and_actual_loaded_identity_visible(self):
        self.pinned_world("--allow-unhealthy", "--left", str(self.star + 1))
        self.assertEqual(self.ui.status[self.star]["status"], "collapsed")
        self.assertEqual(self.ui.status[self.star]["seed"], 0)
        self.assertTrue(self.ui.status[self.star]["selectable"])
        self.assertEqual(self.ui.thumbs[self.star].get_alpha(), 60)
        self.assertEqual(self.world.left["health"], "collapsed")
        self.assertEqual(self.world.left["seed_dir"], "seed_000")
        self.click_target(self.star)
        self.assert_identity(self.ui.message, "star", 0, "collapsed")
        self.ui.notices.clear()
        self.ui.message = "A later unrelated notice must not erase loaded provenance."
        self.assert_checkpoint(self.underboard_text(), 0, "collapsed")
        self.assertEqual(self.world.report(0)["left"]["health"], "collapsed")

    def test_cached_unhealthy_bundle_cannot_bypass_later_healthy_only_policy(self):
        self.pinned_world("--allow-unhealthy", "--left", str(self.star + 1))
        admitted_bundle = self.world.left
        loaded_count = self.load.call_count
        self.world.step(3)
        self.ui.refresh()
        self.world.args.allow_unhealthy = False
        before = self.frozen_state()
        self.click_target(self.star)
        self.assert_frozen(before)
        self.assertIs(self.world.left, admitted_bundle)
        self.assertFalse(self.ui.status[self.star]["selectable"])
        self.assert_identity(self.ui.message, "star", 0, "collapsed")
        self.assertEqual(self.load.call_count, loaded_count)

        self.world.args.allow_unhealthy = True
        self.click_target(self.star)
        self.assertIs(self.world.left, admitted_bundle)
        self.assertTrue(self.ui.status[self.star]["selectable"])
        self.assert_identity(self.ui.message, "star", 0, "collapsed")
        self.assertEqual(self.world.steps, 0)
        self.assertEqual(self.load.call_count, loaded_count)

    def test_failed_clicks_and_number_keys_preserve_match_and_all_rngs(self):
        for duel in (False, True):
            for input_kind in ("mouse", "keyboard"):
                for side, index, seed, health in ((0, self.star, 0, "collapsed"),
                                                (1, self.umbrella, 1, "missing")):
                    with self.subTest(duel=duel, input=input_kind, side=side, health=health):
                        self.pinned_world(*(["--duel"] if duel else []))
                        self.world.step(3)
                        self.ui.refresh()
                        # Keyboard routing must refresh policy even without a prior
                        # side-button click or draw. Mouse routing uses the button.
                        if input_kind == "mouse":
                            self.ui.action("left" if side == 0 else "right")
                        before = self.frozen_state()
                        for _ in range(3):
                            if input_kind == "mouse":
                                self.click_target(index)
                            else:
                                self.key(pygame.K_1 + index, pygame.KMOD_SHIFT if side else 0)
                            self.assert_frozen(before)
                            self.assert_identity(self.ui.message, self.targets[index].stem.split("_", 1)[1], seed, health)

    def test_repeated_draws_do_not_rescan_and_policy_changes_refresh_status(self):
        with patch("arena.target_status", wraps=clash.target_status) as status:
            self.pinned_world()
            initial_calls = status.call_count
            self.assertEqual(initial_calls, len(self.targets))
            with patch("clash.discover_v2_checkpoint", wraps=clash.discover_v2_checkpoint) as discover:
                for _ in range(6):
                    self.ui.draw(dt=.1)
                    self.ui.refresh()
                discover.assert_not_called()
            self.assertEqual(status.call_count, initial_calls)
            self.ui.action("right")
            self.ui.draw(dt=0)
            self.assertEqual(status.call_count, initial_calls + len(self.targets))
            self.assertEqual(self.ui.status[self.star]["seed"], 1)
            right_status = self.ui.status
            self.ui.action("left")
            self.ui.draw(dt=0)
            self.ui.action("right")
            self.ui.draw(dt=0)
            self.assertIs(self.ui.status, right_status)
            self.assertEqual(status.call_count, initial_calls + len(self.targets))
            self.world.args.right_seed = 2
            self.ui.draw(dt=0)
            self.assertEqual(self.ui.status[self.sun]["status"], "ready")
            self.assertEqual(self.ui.status[self.sun]["seed"], 2)
            after_pin = status.call_count
            self.world.args.allow_unhealthy = True
            self.ui.draw(dt=0)
            self.assertEqual(status.call_count, after_pin + len(self.targets))
            after_policy = status.call_count
            for _ in range(4):
                self.ui.draw(dt=.1)
            self.assertEqual(status.call_count, after_policy)

    def test_unchanged_policy_preserves_explicit_status_mutation_for_render_tests(self):
        self.pinned_world()
        entry = self.ui.status[self.heart]
        entry["status"] = "collapsed"
        with patch("arena.target_status", side_effect=AssertionError("unexpected rescan")):
            for _ in range(3):
                self.ui.draw(dt=0)
        self.assertIs(self.ui.status[self.heart], entry)
        self.assertEqual(self.ui.thumbs[self.heart].get_alpha(), 60)

    def test_selection_attempts_refresh_cached_metadata_without_refreshing_every_frame(self):
        self.pinned_world()
        self.ui.status[self.star]["status"] = "ready"
        with patch("arena.target_status", wraps=clash.target_status) as status:
            self.click_target(self.star)
            self.assertEqual(self.ui.status[self.star]["status"], "collapsed")
            self.assertEqual(status.call_count, len(self.targets))
            self.click_target(self.umbrella)
            self.assertEqual(status.call_count, 2 * len(self.targets))
            self.assertEqual(self.world.left["name"], "07_umbrella")
            for _ in range(3):
                self.ui.draw(dt=0)
            self.assertEqual(status.call_count, 2 * len(self.targets))

    def test_equal_side_pins_share_policy_cache_and_lesson_uses_left_pin(self):
        self.pinned_world()
        left_status = self.ui.status
        self.world.args.right_seed = 0
        with patch("arena.target_status", side_effect=AssertionError("same pin rescanned")):
            self.ui.action("right")
            self.ui.draw(dt=0)
        self.assertIs(self.ui.status, left_status)
        self.lesson_world()
        self.ui.side = 1  # Lesson selectors are left-only, including stale UI state.
        self.ui.draw(dt=0)
        self.assertEqual(self.ui.picker_policy(), (1, False))
        self.assertEqual(self.ui.status[self.star]["status"], "ready")
        self.assertEqual(self.ui.status[self.star]["seed"], 1)

    def test_labels_fit_minimum_and_default_widths_without_underboard_overlap(self):
        for width, height in ((860, 740), (1100, 902)):
            for lesson in (False, True):
                with self.subTest(window=(width, height), lesson=lesson):
                    self.new_world("--window-size", str(width), "--allow-unhealthy",
                                   "--left", str(self.star + 1), "--left-seed", "0",
                                   "--right-seed", "1", *(["--lesson"] if lesson else []))
                    self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
                    labels = self.texts()
                    sim = next((text, rect) for text, rect in labels if text.startswith("SIM SEED"))
                    self.assertIn("27", sim[0])
                    heading = next((text, rect) for text, rect in labels if text.startswith("CULTURES"))
                    policy = next((text, rect) for text, rect in labels if "PIN" in text)
                    self.assertIn("PIN 000", policy[0])
                    self.assertFalse(heading[1].colliderect(policy[1]))
                    below = self.texts(self.ui.draw_underboard)
                    self.assertEqual(len(below), 2)
                    self.assertIn("L seed 000 collapsed", below[1][0])
                    if not lesson:
                        self.assertIn("R seed 001 ready", below[1][0])
                    else:
                        self.assertNotIn("R seed", below[1][0])
                    for _, rect in (sim, heading, policy, *below):
                        self.assertTrue(self.ui.window.get_rect().contains(rect))
                    self.assertFalse(below[0][1].colliderect(below[1][1]))
                    self.assertLessEqual(below[1][1].right, self.ui.board.right)
                    self.assertGreaterEqual(below[0][1].top, self.ui.board.bottom)

    def test_lesson_exit_preserves_requested_right_pin_then_loads_healthy_sun(self):
        for action in ("lesson", "duel"):
            with self.subTest(exit=action):
                self.lesson_world()
                self.world.plant_lesson()
                self.world.step(3)
                self.ui.action(action)
                self.assertIsNone(self.world.lesson)
                self.assertIs(self.world.right, self.world.left)
                self.assertEqual(self.world.right["name"], "02_star")
                self.assertEqual(self.world.right["seed_dir"], "seed_001")
                self.assertEqual((self.world.args.left_seed, self.world.args.right_seed), (1, 2))
                self.assertIn("unavailable", self.ui.message.lower())
                self.assert_identity(self.ui.message, "star", 1, "ready")
                self.ui.action("right")
                self.ui.draw(dt=0)
                self.assertEqual(self.ui.status[self.star]["status"], "missing")
                self.assertEqual(self.ui.status[self.sun]["status"], "ready")
                self.assertEqual(self.ui.status[self.sun]["seed"], 2)
                self.assert_checkpoint(self.underboard_text(), 1, "ready")
                self.click_target(self.sun)
                self.assertEqual(self.world.right["name"], "03_sun")
                self.assertEqual(self.world.right["seed_dir"], "seed_002")
                self.assertEqual((self.world.args.left_seed, self.world.args.right_seed), (1, 2))
                self.assert_identity(self.ui.message, "sun", 2, "ready")

    def test_report_and_recipe_use_actual_fallback_then_new_right_checkpoint(self):
        self.lesson_world()
        self.ui.action("duel")
        for select_sun in (False, True):
            with self.subTest(after_selection=select_sun):
                if select_sun:
                    self.key(pygame.K_1 + self.sun, pygame.KMOD_SHIFT)
                self.world.step(20)
                self.assertEqual(self.world.duel.phase, "finished")
                report, recipe = self.world.report(0), self.world.duel_recipe()
                expected_name, expected_seed = ("03_sun", 2) if select_sun else ("02_star", 1)
                self.assertEqual(report["right"]["name"], expected_name)
                self.assertEqual(report["right"]["seed_dir"], f"seed_{expected_seed:03d}")
                self.assertEqual(report["right"]["health"], "ready")
                self.assertIn(f"/{expected_name}/seed_{expected_seed:03d}/", report["right"]["source"])
                self.assertEqual(recipe["models"]["right"],
                                 {"target": expected_name, "checkpoint_seed": expected_seed})
                self.assertEqual(recipe["models"]["left"], {"target": "02_star", "checkpoint_seed": 1})
                self.assertEqual((self.world.args.left_seed, self.world.args.right_seed), (1, 2))
                self.assertEqual(recipe["configuration"]["simulation_seed"], 27)


if __name__ == "__main__":
    unittest.main()
