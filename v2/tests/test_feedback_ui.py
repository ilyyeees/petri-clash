"""Headless SDL integration coverage for presentation-only arena feedback.

Small deterministic, untrained models keep UI assertions fast. This suite does
not measure trained-model quality or native-window/display performance.
"""
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame
import torch

import arena
import clash
from nca import NCA


def synthetic_bundle(target, device, *args, **kwargs):
    model = NCA(channels=8, hidden_size=16).to(device).eval()
    return {"model": model, "channels": 8, "grid_size": 16,
            "channels_last": False, "name": target.stem,
            "health": "ready", "seed_dir": "seed_000", "score": .01,
            "kind": "synthetic-untrained", "source": "ui-test"}


class FeedbackUIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch("arena.ensure_model", side_effect=synthetic_bundle)
        self.loader.start()
        args = arena.parse_args(["--device", "cpu", "--grid-size", "16"])
        self.world = arena.Arena(args, clash.list_targets())
        self.ui = arena.ArenaUI(self.world)
        self.ui.draw(dt=0)

    def tearDown(self):
        pygame.quit()
        self.loader.stop()

    def snapshots(self):
        return [value.clone() for value in (self.world.a, self.world.b,
                                           self.world.owner, self.world.control)]

    def assert_snapshots_equal(self, expected):
        for actual, prior in zip(self.snapshots(), expected):
            self.assertTrue(torch.equal(actual, prior))

    def assert_exact_scores(self):
        expected = self.world.stats()
        self.assertEqual(self.ui.stats_cache, expected)
        key = "territory" if self.world.mode == "hard" else "alive"
        self.assertEqual(self.ui.summary["left"], expected[f"left_{key}"])
        self.assertEqual(self.ui.summary["right"], expected[f"right_{key}"])

    def test_draw_preserves_simulation_and_all_random_generators(self):
        self.world.step(3)
        expected = self.snapshots()
        steps = self.world.steps
        torch_rng = torch.get_rng_state().clone()
        numpy_rng = np.random.get_state()
        python_rng = random.getstate()
        for view in ("organisms", "territory", "pressure"):
            self.ui.action(("view", view))
            for _ in range(3):
                self.ui.draw(dt=1 / 30)
        self.assert_snapshots_equal(expected)
        self.assertEqual(steps, self.world.steps)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
        self.assertEqual(numpy_rng[2:], current_numpy[2:])
        self.assertEqual(python_rng, random.getstate())

    def test_draws_and_fx_do_not_change_replayed_trajectory(self):
        for _ in range(3):
            self.world.step(4)
            self.ui.draw(dt=.04)
            self.ui.action("colors")
            self.ui.action("motion")
        expected = self.snapshots()
        expected_rng = torch.get_rng_state().clone()
        self.ui.action("reset")
        self.world.step(12)
        self.assert_snapshots_equal(expected)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_paused_edits_refresh_exact_counts_before_next_draw(self):
        self.ui.action("pause")
        self.ui.action("clear")
        self.ui.interact(4, 5, 0)
        self.assert_exact_scores()
        self.assertEqual(self.ui.summary["left"], 1)
        self.assertEqual(self.world.steps, 0)
        self.ui.interact(4, 5, 1)
        self.assert_exact_scores()
        self.assertEqual((self.ui.summary["left"], self.ui.summary["right"]), (0, 1))
        self.ui.interact(4, 5)
        self.assert_exact_scores()
        self.assertEqual((self.ui.summary["left"], self.ui.summary["right"]), (0, 0))
        self.assertIn("L -0 / R -1 life", self.ui.effects[-1]["label"])
        self.assertIsNone(self.ui.blend.rendered)
        self.ui.draw(dt=.01)
        np.testing.assert_array_equal(self.ui.blend.rendered, self.world.rgb())

    def test_soft_paused_overlap_is_life_not_ownership(self):
        self.ui.action("mode")
        self.ui.action("clear")
        self.ui.paused = True
        self.ui.interact(5, 5, 0)
        self.ui.interact(5, 5, 1)
        self.assert_exact_scores()
        self.assertEqual((self.ui.summary["left"], self.ui.summary["right"]), (1, 1))
        self.assertEqual(self.ui.stats_cache["contested"], 1)
        self.assertEqual(self.ui.summary["neutral"], 255)
        self.assertEqual(self.ui.summary["metric"], "living cells")
        self.ui.action(("view", "pressure"))
        self.assertEqual(self.ui.view, "organisms")
        self.assertIn("hard mode", self.ui.message)

    def test_reset_clear_mode_and_culture_clear_old_feedback(self):
        for action in ("reset", "clear", "mode", ("target", 0)):
            with self.subTest(action=action):
                self.ui.interact(4, 4, 0)
                self.world.step(2)
                self.ui.draw(dt=.02)
                self.ui.action(action)
                self.assertEqual(self.world.steps, 0)
                self.assertEqual(self.ui.summary["span"], 0)
                self.assertEqual(self.ui.effects, [])
                self.assertTrue(all("planted" not in notice[0] for notice in self.ui.notices))
                self.assertIsNone(self.ui.blend.rendered)
                self.assert_exact_scores()
                self.ui.draw(dt=.001)
                np.testing.assert_array_equal(self.ui.blend.rendered, self.world.rgb(self.ui.team_colors, self.ui.view))

    def test_view_color_mode_changes_snap_maps_without_cross_fading(self):
        self.world.step()
        for action in (("view", "territory"), ("view", "pressure"),
                       "colors", "mode", "motion"):
            self.ui.action(action)
            self.ui.draw(dt=.001)
            np.testing.assert_array_equal(self.ui.blend.rendered,
                self.world.rgb(self.ui.team_colors, self.ui.view))

    def test_repeated_edits_bound_effects_and_draw_time_expires_them(self):
        for i in range(100):
            self.ui.interact(i % 16, (i // 16) % 16, i % 2)
        self.assertEqual(len(self.ui.effects), 24)
        self.assertEqual(len(self.ui.notices), 3)
        self.ui.paused = True
        for _ in range(6):
            self.ui.draw(dt=.25)
        self.assertEqual(self.ui.effects, [])
        self.assertLessEqual(len(self.ui.notices), 3)
        self.assertEqual(self.world.steps, 0)

    def test_paused_draws_preserve_tick_based_trend(self):
        self.world.step(8)
        self.ui.draw(dt=.03)
        self.ui.paused = True
        summary = dict(self.ui.summary)
        for _ in range(10):
            self.ui.draw(dt=.25)
        self.assertEqual(summary, self.ui.summary)
        self.assertEqual(self.ui.summary["span"], 8)

    def test_pause_step_speed_keyboard_and_button_flows(self):
        def key(k):
            self.ui.event(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0))
        key(pygame.K_SPACE)
        self.assertTrue(self.ui.paused)
        for speed in (2, 4, 8, 1):
            key(pygame.K_TAB)
            self.assertEqual(self.ui.speed, speed)
            self.assertEqual(self.world.steps, 0)
        for n in range(1, 4):
            key(pygame.K_n)
            self.assertTrue(self.ui.paused)
            self.assertEqual(self.world.steps, n)
            self.assert_exact_scores()
        self.ui.draw(dt=.01)
        resume = next(rect for rect, action in self.ui.buttons if action == "pause")
        self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=resume.center, button=1))
        self.assertFalse(self.ui.paused)
        key(pygame.K_f)
        self.assertTrue(self.ui.reduced_motion)
        self.ui.draw(dt=.01)
        np.testing.assert_array_equal(self.ui.blend.rendered, self.world.rgb())

    def test_resize_immediately_updates_click_geometry_and_contains_controls(self):
        for width, height in ((200, 200), (1100, 902), (1800, 740), (860, 1200)):
            self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
            window = self.ui.window.get_rect()
            self.assertTrue(window.contains(self.ui.board))
            self.assertTrue(window.contains(self.ui.score_rect))
            self.assertEqual(self.ui.rail_x - self.ui.board.right, 16)
            for rect, action in self.ui.buttons:
                self.assertTrue(window.contains(rect))
                if action in ("pause", "reset", "step", "speed", "mode", "colors", "motion"):
                    self.assertGreaterEqual(rect.left, self.ui.board.left)
                    self.assertLessEqual(rect.right, self.ui.board.right)
                    self.assertLess(rect.bottom, self.ui.board.top)
            with patch.object(self.ui, "interact") as interact:
                self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=self.ui.board.center, button=3))
                interact.assert_called_once_with(8, 8, 1)
                self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(30, 30), button=1))
                self.assertEqual(interact.call_count, 1)

    def test_shift_left_right_and_nonprimary_clicks_are_distinct(self):
        center = self.ui.board.center
        with patch.object(self.ui, "interact") as interact:
            with patch("pygame.key.get_mods", return_value=pygame.KMOD_SHIFT):
                self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=center, button=1))
            interact.assert_called_once_with(8, 8, 0)
            self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=center, button=3))
            self.assertEqual(interact.call_args.args, (8, 8, 1))
            for button in (2, 4, 5):
                self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=center, button=button))
            self.assertEqual(interact.call_count, 2)

    def test_soft_bars_scale_each_side_to_the_whole_board(self):
        self.ui.action("mode")
        self.ui.action("clear")
        self.ui.interact(4, 4, 0)
        self.ui.interact(4, 4, 1)
        # Record the two small colored fills separately from the panel outlines.
        with patch("pygame.draw.rect", wraps=pygame.draw.rect) as rectangles:
            self.ui.draw_score(self.ui.window.get_width())
        backgrounds = self.ui.score_bars
        fills = [pygame.Rect(call.args[2]) for call in rectangles.call_args_list
                 if call.args[1] in (arena.GREEN, arena.CORAL)
                 and any(bar.contains(pygame.Rect(call.args[2])) for bar in backgrounds)]
        self.assertEqual(len(backgrounds), 2)
        self.assertEqual(len(fills), 2)
        for background, fill in zip(backgrounds, fills):
            self.assertTrue(self.ui.score_rect.contains(background))
            self.assertEqual(fill.width, round(background.width / self.world.size ** 2))
            self.assertEqual(fill.x, background.x)
            self.assertLess(fill.width, background.width / 2)

    def test_large_score_places_metric_below_number(self):
        # Exercise a valid 256-square board's maximum text count without an
        # expensive simulation. This test isolates scoreboard typography.
        self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=860, h=740))
        self.ui.stats_cache = {**self.ui.stats_cache, "grid_size": 256,
                               "left_territory": 65_536, "right_territory": 0}
        self.ui.summary = self.ui.score.update(self.ui.stats_cache)
        with patch.object(self.world, "size", 256), \
                patch.object(self.ui, "text", wraps=self.ui.text) as text:
            self.ui.draw_score(860)
        calls = [call.args for call in text.call_args_list]
        number = next(args for args in calls if args[0] == "65,536")
        metric = next(args for args in calls if args[0] == "100.0% held")
        number_pixels = self.ui.fonts[number[3]].render(number[0], True, arena.TEXT).get_bounding_rect()
        metric_pixels = self.ui.fonts[metric[3]].render(metric[0], True, arena.TEXT).get_bounding_rect()
        self.assertGreater(metric[2] + metric_pixels.top, number[2] + number_pixels.bottom)


if __name__ == "__main__":
    unittest.main()
