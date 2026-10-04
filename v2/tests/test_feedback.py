import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feedback import FrameBlend, ScoreTracker


def stats(step=0, left=10, right=5, mode="hard", size=16, **overrides):
    result = {"step": step, "mode": mode, "grid_size": size,
              "left_alive": left, "right_alive": right,
              "left_territory": left if mode == "hard" else None,
              "right_territory": right if mode == "hard" else None,
              "contested": 0, "finite": True}
    return {**result, **overrides}


class FrameBlendTests(unittest.TestCase):
    def setUp(self):
        self.black = np.zeros((4, 6, 3), dtype=np.uint8)
        self.white = np.full_like(self.black, 255)

    def test_first_frame_and_reset_snap_without_aliasing(self):
        blend = FrameBlend()
        result = blend.update(self.black, .02, "land")
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(blend.rendered.dtype, np.float32)
        self.assertEqual(result.shape, self.black.shape)
        self.assertFalse(np.shares_memory(result, self.black))
        self.assertFalse(np.shares_memory(blend.rendered, self.black))
        self.assertTrue(np.array_equal(blend.reset(self.white, "life"), self.white))
        self.assertTrue(np.array_equal(blend.rendered, self.white))
        self.assertIsNone(blend.reset())
        self.assertIsNone(blend.rendered)
        self.assertTrue(np.array_equal(blend.update(self.black, .01), self.black))

    def test_easing_is_time_based_and_frame_rate_invariant(self):
        frames = []
        for fps in (30, 60, 120):
            blend = FrameBlend()
            blend.update(self.black, 0)
            for _ in range(fps // 2):
                blend.update(self.white, 1 / fps)
            frames.append(blend.rendered.copy())
        expected = 255 * (1 - math.exp(-.5 / .1))
        for result in frames:
            np.testing.assert_allclose(result, expected, atol=0.0002)
        np.testing.assert_allclose(frames[0], frames[1], atol=0.0002)

    def test_draws_keep_smoothing_without_new_target_and_do_not_mutate_input(self):
        blend = FrameBlend()
        blend.update(self.black, 0)
        target = self.white.copy()
        early = blend.update(target, .01)
        later = blend.update(target, .01)
        self.assertTrue(np.all(later > early))
        self.assertTrue(np.all(later < target))
        np.testing.assert_array_equal(target, self.white)
        later[:] = 0
        self.assertTrue(np.all(blend.rendered > 0))

    def test_changes_to_view_mode_grid_color_or_epoch_snap(self):
        blend = FrameBlend()
        keys = [(0, "hard", "land", 6, False), (0, "hard", "life", 6, False),
                (0, "soft", "life", 6, False), (0, "soft", "life", 6, True),
                (0, "soft", "life", 7, True), (1, "soft", "life", 7, True)]
        for key in keys:
            blend.reset(self.black, "other")
            np.testing.assert_array_equal(blend.update(self.white, .001, key), self.white)
        small = np.full((2, 3, 3), 123, dtype=np.uint8)
        np.testing.assert_array_equal(blend.update(small, .001, keys[-1]), small)

    def test_long_stalls_and_invalid_timing_do_not_jump_or_poison(self):
        blend = FrameBlend()
        blend.update(self.black, 0)
        for elapsed in (-1, float("nan"), float("inf")):
            np.testing.assert_array_equal(blend.update(self.white, elapsed), self.black)
        result = blend.update(self.white, 200)
        expected = 255 * (1 - math.exp(-1))
        np.testing.assert_allclose(result, round(expected), atol=0)

    def test_invalid_shape_dtype_and_configuration(self):
        for target in ([1, 2, 3], self.black.astype(float)):
            with self.assertRaises(TypeError):
                FrameBlend().update(target, .02)
        for target in (np.zeros((4, 4), dtype=np.uint8), np.zeros((0, 3, 3), dtype=np.uint8),
                       np.zeros((4, 4, 4), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                FrameBlend().update(target, .02)
        for options in ({"tau": 0}, {"tau": float("nan")}, {"max_elapsed": -1}):
            with self.assertRaises(ValueError):
                FrameBlend(**options)


class ScoreTrackerTests(unittest.TestCase):
    def test_hard_score_uses_exact_ownership_not_living_tissue(self):
        tracker = ScoreTracker()
        state = stats(left=19, right=7, left_alive=1, right_alive=80)
        original = copy.deepcopy(state)
        score = tracker.update(state)
        self.assertEqual((score["left"], score["right"], score["lead"]), (19, 7, 12))
        self.assertEqual(score["metric"], "owned cells")
        self.assertEqual(score["leader"], "left")
        self.assertEqual(score["neutral"], 230)
        self.assertAlmostEqual(sum(score["percent"].values()), 100)
        self.assertIn("12 more owned cells", score["reason"])
        self.assertEqual(state, original)

    def test_soft_null_territory_and_overlap_are_supported(self):
        score = ScoreTracker().update(stats(left=60, right=70, mode="soft", size=10, contested=40))
        self.assertEqual(score["metric"], "living cells")
        self.assertEqual(score["leader"], "right")
        self.assertEqual(score["lead"], -10)
        self.assertEqual(score["neutral"], 10)
        self.assertEqual(score["percent"], {"left": 60, "right": 70, "neutral": 10})
        self.assertIn("Independent growth", score["status"])

    def test_repeated_paused_draws_do_not_change_or_age_trend(self):
        tracker = ScoreTracker()
        tracker.update(stats(step=0))
        expected = tracker.update(stats(step=8, left=18, right=3))
        for _ in range(100):
            self.assertEqual(tracker.update(stats(step=8, left=18, right=3)), expected)
        self.assertEqual(expected["delta"], {"left": 8, "right": -2, "lead": 10})
        self.assertEqual(expected["span"], 8)
        self.assertEqual(len(tracker._samples), 2)

    def test_tick_window_at_one_and_eight_steps_per_render(self):
        for stride, span in ((1, 30), (8, 32)):
            with self.subTest(stride=stride):
                tracker = ScoreTracker()
                for step in range(0, 81, stride):
                    result = tracker.update(stats(step=step, left=10 + step, right=5, size=20))
                self.assertEqual(result["span"], span)
                self.assertEqual(result["delta"]["left"], span)
                self.assertEqual(result["delta"]["right"], 0)
                self.assertIn(f"over {span} ticks", result["trend"])

    def test_paused_edit_refreshes_counts_without_advancing_tick_window(self):
        tracker = ScoreTracker()
        tracker.update(stats(step=0))
        tracker.update(stats(step=8, left=18))
        changed = tracker.update(stats(step=8, left=12))
        self.assertEqual(changed["left"], 12)
        self.assertEqual(changed["span"], 8)
        self.assertEqual(changed["delta"]["left"], 2)
        self.assertEqual(len(tracker._samples), 2)
        self.assertEqual(changed, tracker.update(stats(step=8, left=12)))

    def test_explicit_reset_and_new_epoch_at_same_step_clear_history(self):
        tracker = ScoreTracker()
        tracker.update(stats(step=0), epoch=0)
        tracker.update(stats(step=8, left=20), epoch=0)
        fresh = tracker.update(stats(step=8, left=4), epoch=1)
        self.assertEqual(fresh["span"], 0)
        self.assertEqual(fresh["delta"], {"left": 0, "right": 0, "lead": 0})
        tracker.update(stats(step=16, left=30), epoch=1)
        tracker.reset()
        self.assertEqual(tracker.update(stats(step=16, left=1))["span"], 0)

    def test_mode_grid_or_decreasing_step_resets_history(self):
        for next_state in (stats(step=2), stats(step=9, mode="soft"), stats(step=9, size=20)):
            with self.subTest(next_state=next_state):
                tracker = ScoreTracker()
                tracker.update(stats(step=0))
                tracker.update(stats(step=8, left=30))
                result = tracker.update(next_state)
                self.assertEqual(result["span"], 0)
                self.assertEqual(result["delta"]["left"], 0)

    def test_ties_empty_and_unclaimed_never_declare_match_over(self):
        cases = (stats(left=5, right=5), stats(left=0, right=0),
                 stats(left=0, right=0, left_alive=4), stats(left=5, right=0))
        for case in cases:
            result = ScoreTracker().update(case)
            all_text = " ".join(result[key].lower() for key in ("headline", "reason", "trend", "status"))
            for incorrect in ("winner", "eliminated", "defeated", "killed"):
                self.assertNotIn(incorrect, all_text)
        tied = ScoreTracker().update(cases[0])
        self.assertEqual(tied["leader"], None)
        self.assertEqual(tied["headline"], "LEVEL")
        empty = ScoreTracker().update(cases[1])
        self.assertEqual(empty["neutral"], 256)
        self.assertEqual(empty["percent"]["neutral"], 100)
        self.assertIn("no territory claimed", ScoreTracker().update(cases[2])["status"])

    def test_lost_cells_do_not_imply_combat_or_capture(self):
        tracker = ScoreTracker()
        tracker.update(stats(left=40, right=20))
        result = tracker.update(stats(step=30, left=30, right=20, left_alive=10))
        self.assertEqual(result["delta"]["left"], -10)
        all_text = " ".join(result[key].lower() for key in ("headline", "reason", "trend", "status"))
        for inferred_cause in ("attack", "damage", "captur", "kill", "combat"):
            self.assertNotIn(inferred_cause, all_text)

    def test_nonfinite_warning_and_invalid_inputs(self):
        self.assertIn("Non-finite", ScoreTracker().update(stats(finite=False))["status"])
        for state in (stats(mode="unknown"), stats(size=0), stats(step=-1), stats(left_territory=None)):
            with self.assertRaises(ValueError):
                ScoreTracker().update(state)
        for window in (0, -1, .5, True):
            with self.assertRaises(ValueError):
                ScoreTracker(window_ticks=window)


if __name__ == "__main__":
    unittest.main()
