"""Pure-Python duel rules; these tests require neither Torch nor rendering."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duel import DuelConfig, DuelRound


class DuelConfigTests(unittest.TestCase):
    def test_defaults_and_config_immutability(self):
        config = DuelConfig()
        self.assertEqual((config.total_ticks, config.warmup_ticks), (600, 60))
        with self.assertRaises(FrozenInstanceError):
            config.total_ticks = 10

    def test_invalid_config_requires_actual_integers_and_a_scoring_window(self):
        for total in (0, -1, True, 1.0, 2.5, "600", None, float("nan"), float("inf")):
            with self.subTest(total=total), self.assertRaises(ValueError):
                DuelConfig(total_ticks=total, warmup_ticks=0)
        for warmup in (-1, True, 0.0, .5, "60", None, float("nan"), float("inf"), 5, 6):
            with self.subTest(warmup=warmup), self.assertRaises(ValueError):
                DuelConfig(total_ticks=5, warmup_ticks=warmup)
        self.assertEqual(DuelConfig(1, 0).total_ticks, 1)

    def test_invalid_round_configuration(self):
        with self.assertRaises(TypeError):
            DuelRound(config={"total_ticks": 5})
        for mode in (None, "HARD", "unknown", 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                DuelRound(mode=mode)
        for capacity in (0, -1, True, 1.0, "16", float("inf")):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                DuelRound(board_cells=capacity)


class DuelRoundTests(unittest.TestCase):
    def test_initial_snapshot_is_unscored_and_unknown_not_a_draw(self):
        duel = DuelRound()
        state = duel.snapshot()
        self.assertEqual(state["phase"], "warmup")
        self.assertEqual(state["tick"], 0)
        self.assertEqual(state["remaining_ticks"], 600)
        self.assertEqual(state["warmup_remaining_ticks"], 60)
        self.assertEqual(state["scored_ticks"], 0)
        self.assertEqual(state["scores"], {"left": 0, "right": 0})
        self.assertEqual(state["averages"], {"left": None, "right": None})
        self.assertEqual(state["current"], {"left": None, "right": None})
        self.assertIsNone(state["winner"])
        self.assertIsNone(state["final"])
        self.assertIsNone(duel.result)
        self.assertFalse(duel.finished)
        self.assertTrue(state["valid"])

    def test_default_window_excludes_exactly_60_then_scores_exactly_540(self):
        duel = DuelRound()
        for tick in range(1, 61):
            state = duel.record_tick(tick, 10000, 0)
            self.assertEqual(state["scores"], {"left": 0, "right": 0})
            self.assertEqual(state["scored_ticks"], 0)
            self.assertEqual(state["phase"], "warmup" if tick < 60 else "scoring")
        self.assertEqual(state["warmup_remaining_ticks"], 0)
        for tick in range(61, 601):
            state = duel.record_tick(tick, 2, 3)
        self.assertEqual(state["phase"], "finished")
        self.assertEqual(state["scored_ticks"], 540)
        self.assertEqual(state["scores"], {"left": 1080, "right": 1620})
        self.assertEqual(state["averages"], {"left": 2.0, "right": 3.0})
        self.assertEqual(state["final"], {"left": 2, "right": 3})
        self.assertEqual(state["winner"], "right")
        self.assertEqual(state["remaining_ticks"], 0)
        self.assertEqual(duel.result, state)

    def test_zero_warmup_one_tick_and_last_tick_inclusive(self):
        duel = DuelRound(DuelConfig(1, 0))
        self.assertEqual(duel.phase, "scoring")
        state = duel.record_tick(1, 0, 1)
        self.assertEqual(state["scored_ticks"], 1)
        self.assertEqual(state["scores"], {"left": 0, "right": 1})
        self.assertEqual(state["winner"], "right")

    def test_score_is_area_time_not_endpoint_or_average_of_render_samples(self):
        duel = DuelRound(DuelConfig(4, 1))
        duel.record_tick(1, 0, 999)
        duel.record_tick(2, 10, 0)
        duel.record_tick(3, 10, 0)
        state = duel.record_tick(4, 0, 19)
        self.assertEqual(state["scores"], {"left": 20, "right": 19})
        self.assertAlmostEqual(state["averages"]["left"], 20 / 3)
        self.assertEqual(state["final"], {"left": 0, "right": 19})
        self.assertEqual(state["winner"], "left")

    def test_clip_steps_exact_horizon_at_all_ui_speeds(self):
        outputs = []
        for speed in (1, 2, 4, 8, 100):
            duel = DuelRound(DuelConfig(11, 3))
            simulated = []
            while not duel.finished:
                for _ in range(duel.clip_steps(speed)):
                    tick = duel.tick + 1
                    simulated.append(tick)
                    duel.record_tick(tick, tick, 12 - tick)
            self.assertEqual(simulated, list(range(1, 12)))
            self.assertEqual(duel.clip_steps(speed), 0)
            outputs.append(duel.result)
        self.assertTrue(all(output == outputs[0] for output in outputs))

    def test_clip_steps_validation_and_zero_are_atomic(self):
        duel = DuelRound()
        original = duel.snapshot()
        self.assertEqual(duel.clip_steps(0), 0)
        self.assertEqual(duel.clip_steps(999), 600)
        for count in (-1, True, 1.0, "8", None, float("nan"), float("inf")):
            with self.subTest(count=count), self.assertRaises(ValueError):
                duel.clip_steps(count)
            self.assertEqual(duel.snapshot(), original)

    def test_duplicate_skipped_backwards_and_noninteger_ticks_are_rejected_atomically(self):
        duel = DuelRound(DuelConfig(5, 0))
        original = duel.snapshot()
        for tick in (0, 2, 5, -1, True, 1.0, "1", None, float("nan"), float("inf")):
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                duel.record_tick(tick, 1, 0)
            self.assertEqual(duel.snapshot(), original)
        duel.record_tick(1, 2, 1)
        original = duel.snapshot()
        for tick in (1, 3, 0):
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                duel.record_tick(tick, 99, 99)
            self.assertEqual(duel.snapshot(), original)
        duel.record_tick(2, 3, 1)
        self.assertEqual(duel.snapshot()["scores"], {"left": 5, "right": 2})

    def test_terminal_snapshot_is_frozen_and_exports_do_not_alias(self):
        duel = DuelRound(DuelConfig(1, 0))
        snapshot = duel.record_tick(1, 9, 8)
        expected = duel.result
        snapshot["scores"]["left"] = 0
        snapshot["current"]["left"] = 0
        self.assertEqual(snapshot["final"]["left"], 9)
        snapshot["final"]["left"] = 0
        snapshot["averages"]["left"] = 0
        snapshot["winner"] = "right"
        exported = duel.result
        exported["scores"]["right"] = 100
        self.assertEqual(duel.result, expected)
        for tick, finite in ((1, True), (2, True), (2, False), (100, True)):
            with self.subTest(tick=tick, finite=finite), self.assertRaises(ValueError):
                duel.record_tick(tick, 0, 999, finite=finite)
            self.assertEqual(duel.result, expected)

    def test_reset_replays_scoring_and_preserves_old_exports(self):
        duel = DuelRound(DuelConfig(2, 1), board_cells=10)
        config = duel.config
        initial = duel.snapshot()
        duel.record_tick(1, 9, 1)
        first = duel.record_tick(2, 1, 9)
        saved = json.dumps(first, sort_keys=True)
        duel.reset()
        self.assertEqual(duel.snapshot(), initial)
        self.assertIs(duel.config, config)
        self.assertEqual(duel.board_cells, 10)
        duel.record_tick(1, 9, 1)
        self.assertEqual(duel.record_tick(2, 1, 9), first)
        self.assertEqual(json.dumps(first, sort_keys=True), saved)

    def test_invalid_counts_and_finite_flags_reject_without_advancing(self):
        duel = DuelRound(DuelConfig(2, 0), board_cells=10)
        duel.record_tick(1, 2, 1)
        expected = duel.snapshot()
        bad_counts = (-1, True, 2.0, .5, "2", None, float("nan"), float("inf"), 11)
        for bad in bad_counts:
            for left, right in ((bad, 0), (0, bad)):
                with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                    duel.record_tick(2, left, right)
                self.assertEqual(duel.snapshot(), expected)
        with self.assertRaises(ValueError):
            duel.record_tick(2, 6, 5)
        for finite in (None, 0, 1, "true", float("nan")):
            with self.subTest(finite=finite), self.assertRaises(ValueError):
                duel.record_tick(2, 2, 1, finite=finite)
            self.assertEqual(duel.snapshot(), expected)
        self.assertEqual(duel.record_tick(2, 2, 1)["winner"], "left")

    def test_nonfinite_state_ends_invalid_during_warmup_scoring_or_endpoint(self):
        for invalid_tick in (1, 3, 4):
            duel = DuelRound(DuelConfig(4, 1))
            for tick in range(1, invalid_tick):
                duel.record_tick(tick, 10, 2)
            before = duel.snapshot()
            state = duel.record_tick(invalid_tick, float("nan"), None, finite=False)
            self.assertEqual(state["phase"], "invalid")
            self.assertEqual(state["tick"], invalid_tick)
            self.assertTrue(state["finished"])
            self.assertFalse(state["valid"])
            self.assertEqual(state["scores"], before["scores"])
            self.assertEqual(state["scored_ticks"], before["scored_ticks"])
            self.assertEqual(state["current"], {"left": None, "right": None})
            self.assertIsNone(state["final"])
            self.assertIsNone(state["winner"])
            self.assertIn("Non-finite", state["invalid_reason"])
            self.assertEqual(duel.clip_steps(8), 0)
            with self.assertRaises(ValueError):
                duel.record_tick(invalid_tick + 1, 9, 0)
            self.assertEqual(duel.result, state)
            duel.reset()
            self.assertTrue(duel.snapshot()["valid"])
            self.assertFalse(duel.finished)

    def test_exact_ties_including_empty_board_are_draws_only_at_endpoint(self):
        for samples in (((0, 0), (0, 0)), ((1, 3), (3, 1)), ((7, 7), (7, 7))):
            duel = DuelRound(DuelConfig(2, 0))
            self.assertIsNone(duel.record_tick(1, *samples[0])["winner"])
            state = duel.record_tick(2, *samples[1])
            self.assertEqual(state["winner"], "draw")
            self.assertEqual(state["scores"]["left"], state["scores"]["right"])

    def test_one_cell_tick_margin_is_not_rounded_or_relative_tolerance_tied(self):
        count = 2**60
        duel = DuelRound(DuelConfig(2, 0))
        duel.record_tick(1, count, count)
        state = duel.record_tick(2, count + 1, count)
        self.assertEqual(state["scores"]["left"] - state["scores"]["right"], 1)
        self.assertEqual(state["averages"]["left"], state["averages"]["right"])
        self.assertEqual(state["winner"], "left")
        self.assertIs(type(state["scores"]["left"]), int)

    def test_soft_mode_counts_overlap_and_are_labeled_living_not_territory(self):
        duel = DuelRound(DuelConfig(1, 0), mode="soft", board_cells=10)
        state = duel.record_tick(1, 10, 10)
        self.assertEqual(state["metric"], "living cells")
        self.assertEqual(state["final"], {"left": 10, "right": 10})
        self.assertEqual(state["winner"], "draw")
        hard = DuelRound(DuelConfig(1, 0), board_cells=10)
        self.assertEqual(hard.snapshot()["metric"], "held cells")
        with self.assertRaises(ValueError):
            hard.record_tick(1, 10, 10)

    def test_swapping_sides_swaps_scores_and_winner_symmetrically(self):
        rng = random.Random(17)
        for mode in ("hard", "soft"):
            duel = DuelRound(DuelConfig(40, 7), mode=mode, board_cells=100)
            swapped = DuelRound(duel.config, mode=mode, board_cells=100)
            for tick in range(1, 41):
                left, right = rng.randrange(51), rng.randrange(51)
                state = duel.record_tick(tick, left, right)
                other = swapped.record_tick(tick, right, left)
                for key in ("scores", "averages", "current"):
                    self.assertEqual(state[key]["left"], other[key]["right"])
                    self.assertEqual(state[key]["right"], other[key]["left"])
                self.assertEqual(state["winner"], {"left": "right", "right": "left",
                                                    "draw": "draw", None: None}[other["winner"]])
            self.assertEqual(state["final"]["left"], other["final"]["right"])

    def test_snapshot_is_json_safe_in_all_phases_and_does_not_change_state_or_rng(self):
        duel = DuelRound(DuelConfig(3, 1))
        random_state = random.getstate()
        states = [duel.snapshot()]
        states.extend(duel.record_tick(tick, tick, 3 - tick) for tick in range(1, 4))
        duel.reset()
        states.append(duel.record_tick(1, None, None, finite=False))
        for state in states:
            self.assertEqual(json.loads(json.dumps(state, allow_nan=False)), state)
        expected = duel.snapshot()
        for _ in range(20):
            self.assertEqual(duel.snapshot(), expected)
        self.assertEqual(random.getstate(), random_state)

    def test_extreme_integer_scores_remain_exact_when_average_cannot_be_displayed(self):
        enormous = 10**400
        duel = DuelRound(DuelConfig(1, 0))
        state = duel.record_tick(1, enormous, enormous - 1)
        self.assertEqual(state["winner"], "left")
        self.assertEqual(state["scores"]["left"], enormous)
        self.assertEqual(state["averages"], {"left": None, "right": None})
        json.dumps(state, allow_nan=False)

    def test_import_has_no_torch_numpy_or_pygame_dependency(self):
        source = Path(__file__).resolve().parents[1]
        code = (f"import sys; sys.path.insert(0, {str(source)!r}); import duel; "
                "assert not {'torch', 'numpy', 'pygame'} & sys.modules.keys(); "
                "assert duel.DuelRound().snapshot()['total_ticks'] == 600")
        completed = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
