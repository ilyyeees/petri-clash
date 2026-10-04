"""Deterministic lesson rules and crater geometry, without Torch or rendering."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lesson import CraterPlan, Lesson, LessonConfig, choose_crater


class LessonConfigTests(unittest.TestCase):
    def test_defaults_and_frozen_normalized_integers(self):
        config = LessonConfig()
        self.assertEqual((config.grow_ticks, config.recovery_ticks, config.target_num,
                          config.target_den, config.hold_ticks), (160, 160, 9, 10, 24))
        with self.assertRaises(FrozenInstanceError):
            config.grow_ticks = 1
        config = LessonConfig(*(np.int64(value) for value in (1, 2, 1, 1, 2)))
        self.assertTrue(all(type(value) is int for value in vars(config).values()))

    def test_config_requires_positive_integers_and_possible_target_hold(self):
        for name in ("grow_ticks", "recovery_ticks", "target_num", "target_den", "hold_ticks"):
            for value in (0, -1, True, np.bool_(True), 1.0, "1", None, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    LessonConfig(**{name: value})
        with self.assertRaises(ValueError):
            LessonConfig(target_num=11, target_den=10)
        with self.assertRaises(ValueError):
            LessonConfig(recovery_ticks=7, hold_ticks=8)
        self.assertEqual(LessonConfig(1, 1, 1, 1, 1).hold_ticks, 1)

    def test_controller_configuration(self):
        for config in ({}, 1, False):
            with self.subTest(config=config), self.assertRaises(TypeError):
                Lesson(config)
        for cells in (0, -1, True, 1.0, "16", float("inf")):
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                Lesson(board_cells=cells)
        self.assertIs(type(Lesson(board_cells=np.int64(16)).board_cells), int)


class CraterPlanTests(unittest.TestCase):
    def test_valid_plan_is_frozen_exact_and_normalized(self):
        plan = CraterPlan(*(np.int64(value) for value in (0, 1, 2, 20, 5)))
        self.assertEqual(plan.remaining, 15)
        self.assertTrue(all(type(value) is int for value in vars(plan).values()))
        with self.assertRaises(FrozenInstanceError):
            plan.radius = 3
        self.assertEqual(CraterPlan(0, 0, 1, 5, 3).remaining, 2)

    def test_plan_rejects_invalid_geometry_counts_and_unsafe_damage(self):
        defaults = {"x": 0, "y": 0, "radius": 1, "baseline": 20, "removed": 5}
        for name in defaults:
            for value in (-1, True, np.bool_(True), 1.0, "1", None, float("nan")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    CraterPlan(**(defaults | {name: value}))
        for baseline, removed in ((0, 0), (1, 1), (20, 0), (20, 4), (20, 13), (20, 20), (20, 21)):
            with self.subTest(baseline=baseline, removed=removed), self.assertRaises(ValueError):
                CraterPlan(0, 0, 1, baseline, removed)
        with self.assertRaises(ValueError):
            CraterPlan(0, 0, 0, 20, 5)

    def test_smallest_radius_and_inclusive_circle(self):
        mask = np.ones((5, 5), dtype=bool)
        plan = choose_crater(mask)
        self.assertEqual(plan, CraterPlan(2, 2, 2, 25, 13))
        yy, xx = np.indices(mask.shape)
        removed = int(np.count_nonzero(mask & ((xx - plan.x) ** 2 + (yy - plan.y) ** 2 <= plan.radius ** 2)))
        self.assertEqual(removed, plan.removed)
        self.assertLess(4 * np.count_nonzero((xx - 2) ** 2 + (yy - 2) ** 2 <= 1), plan.baseline)

    def test_exact_lower_and_upper_damage_bounds_are_inclusive(self):
        lower = np.zeros((7, 7), dtype=bool)
        for x, y in ((3, 3), (5, 3), (2, 5), (2, 1)):
            lower[y, x] = True
        self.assertEqual(choose_crater(lower), CraterPlan(3, 3, 1, 4, 1))
        upper = np.zeros((1, 7), dtype=bool)
        upper[0, [0, 2, 3, 4, 6]] = True
        self.assertEqual(choose_crater(upper), CraterPlan(3, 0, 1, 5, 3))

    def test_centroid_rounds_half_to_even_in_both_directions_and_axes(self):
        for columns, center in (([0, 2, 3, 5], 2), ([1, 3, 4, 6], 4)):
            mask = np.zeros((1, 7), dtype=bool)
            mask[0, columns] = True
            self.assertEqual(choose_crater(mask), CraterPlan(center, 0, 1, 4, 2))
            self.assertEqual(choose_crater(mask.T), CraterPlan(0, center, 1, 4, 2))

    def test_empty_all_dead_and_no_safe_radius_have_no_fallback(self):
        for mask in (np.zeros((0, 0), dtype=bool), np.zeros((0, 4), dtype=bool),
                     np.zeros((8, 8), dtype=bool), np.ones((1, 1), dtype=bool)):
            with self.subTest(shape=mask.shape):
                self.assertIsNone(choose_crater(mask))
        ring = np.zeros((3, 3), dtype=bool)
        ring[0, 0] = ring[0, 2] = ring[2, 0] = ring[2, 2] = True
        self.assertIsNone(choose_crater(ring))
        # A radius-zero center could remove 1/3, but the lesson starts at r=1.
        self.assertIsNone(choose_crater(np.ones((1, 3), dtype=bool)))

    def test_requires_boolean_two_dimensional_numpy_mask(self):
        for mask in ([[True]], None, True):
            with self.subTest(mask=mask), self.assertRaises(TypeError):
                choose_crater(mask)
        for mask in (np.ones((3, 3)), np.ones((3, 3), dtype=int), np.array(True),
                     np.ones(3, dtype=bool), np.ones((1, 3, 3), dtype=bool)):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                choose_crater(mask)

    def test_mask_views_readonly_inputs_and_rng_are_unchanged(self):
        mask = np.ones((5, 10), dtype=bool)[:, ::2]
        mask.setflags(write=False)
        before = mask.copy()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        for _ in range(5):
            self.assertEqual(choose_crater(mask), CraterPlan(2, 2, 2, 25, 13))
        np.testing.assert_array_equal(mask, before)
        self.assertEqual(random.getstate(), python_rng)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_rng[0])
        np.testing.assert_array_equal(after[1], numpy_rng[1])
        self.assertEqual(after[2:], numpy_rng[2:])


class LessonTests(unittest.TestCase):
    def grown(self, config=None, baseline=10, board_cells=None):
        lesson = Lesson(config or LessonConfig(2, 5, 9, 10, 2), board_cells=board_cells)
        lesson.start_seed()
        for tick in range(1, lesson.config.grow_ticks + 1):
            lesson.record_tick(tick, baseline)
        return lesson

    def recovering(self, config=None, baseline=10, removed=5, board_cells=None):
        lesson = self.grown(config, baseline, board_cells)
        lesson.set_crater(CraterPlan(2, 2, 1, baseline, removed))
        lesson.apply_damage(baseline - removed)
        lesson.watch()
        return lesson

    def assert_atomic(self, lesson, operation, error=ValueError):
        before = lesson.snapshot()
        with self.assertRaises(error):
            operation()
        self.assertEqual(lesson.snapshot(), before)

    def test_initial_snapshot_and_every_successful_stage(self):
        lesson = Lesson(LessonConfig(2, 5, 9, 10, 2))
        state = lesson.snapshot()
        self.assertEqual(state["phase"], "seed")
        self.assertEqual((state["tick"], state["growth_ticks"], state["recovery_ticks"]), (0, 0, 0))
        for key in ("current", "baseline", "removed", "remaining_after_cut", "target", "plan", "reason"):
            self.assertIsNone(state[key])
        self.assertEqual(state["metric"], "living cells")
        self.assertEqual((state["hold"], state["required_hold"]), (0, 2))
        self.assertFalse(lesson.can_step)
        self.assertFalse(lesson.finished)
        self.assertIsNone(lesson.result)
        self.assertEqual(lesson.start_seed()["phase"], "grow")
        self.assertTrue(lesson.can_step)
        self.assertEqual(lesson.record_tick(1, 7)["current"], 7)
        self.assertIsNone(lesson.snapshot()["baseline"])
        state = lesson.record_tick(2, 10)
        self.assertEqual((state["phase"], state["baseline"], state["target"]), ("damage", 10, 9))
        self.assertFalse(lesson.can_step)
        state = lesson.set_crater(CraterPlan(2, 2, 1, 10, 5))
        self.assertEqual((state["removed"], state["current"]), (5, 10))
        self.assertIsNone(state["remaining_after_cut"])
        self.assertEqual(state["plan"], {"x": 2, "y": 2, "radius": 1, "baseline": 10, "removed": 5, "remaining": 5})
        state = lesson.apply_damage(5)
        self.assertEqual((state["phase"], state["current"], state["remaining_after_cut"]), ("injured", 5, 5))
        self.assertEqual((state["tick"], state["hold"]), (2, 0))
        self.assertEqual(lesson.watch()["phase"], "recover")
        state = lesson.record_tick(3, 9)
        self.assertEqual((state["phase"], state["hold"]), ("recover", 1))
        state = lesson.record_tick(4, 12)
        self.assertEqual((state["phase"], state["hold"]), ("complete", 2))
        self.assertEqual((state["growth_ticks"], state["recovery_ticks"]), (2, 2))
        self.assertTrue(state["valid"])
        self.assertTrue(lesson.finished)
        self.assertFalse(lesson.can_step)
        self.assertEqual(lesson.result, state)

    def test_default_growth_has_exact_160_tick_boundary(self):
        lesson = Lesson()
        lesson.start_seed()
        for tick in range(1, 161):
            state = lesson.record_tick(tick, tick)
            self.assertEqual(state["phase"], "grow" if tick < 160 else "damage")
            self.assertEqual(state["growth_ticks"], tick)
        self.assertEqual((state["baseline"], state["target"]), (160, 144))
        self.assertEqual(lesson.clip_steps(10000), 0)
        self.assert_atomic(lesson, lambda: lesson.record_tick(161, 160))

    def test_action_gates_reject_out_of_order_and_repeated_actions(self):
        lesson = Lesson(LessonConfig(1, 2, 9, 10, 1))
        plan = CraterPlan(2, 2, 1, 10, 5)
        for operation in (lambda: lesson.set_crater(plan), lambda: lesson.apply_damage(5),
                          lesson.watch, lambda: lesson.record_tick(1, 10)):
            self.assert_atomic(lesson, operation)
        lesson.start_seed()
        self.assert_atomic(lesson, lesson.start_seed)
        self.assert_atomic(lesson, lambda: lesson.set_crater(plan))
        self.assert_atomic(lesson, lesson.watch)
        lesson.record_tick(1, 10)
        self.assert_atomic(lesson, lambda: lesson.apply_damage(5))
        self.assert_atomic(lesson, lesson.start_seed)
        self.assert_atomic(lesson, lesson.watch)
        self.assert_atomic(lesson, lambda: lesson.record_tick(2, 10))
        self.assert_atomic(lesson, lambda: lesson.set_crater({}), TypeError)
        self.assert_atomic(lesson, lambda: lesson.set_crater(CraterPlan(2, 2, 1, 12, 6)))
        lesson.set_crater(plan)
        self.assert_atomic(lesson, lambda: lesson.set_crater(plan))
        self.assert_atomic(lesson, lambda: lesson.set_crater(None))
        lesson.apply_damage(5)
        self.assert_atomic(lesson, lambda: lesson.apply_damage(5))
        self.assert_atomic(lesson, lambda: lesson.record_tick(2, 9))
        lesson.watch()
        self.assert_atomic(lesson, lesson.watch)
        self.assert_atomic(lesson, lambda: lesson.apply_damage(5))
        lesson.record_tick(2, 9)
        self.assertTrue(lesson.finished)

    def test_cut_mismatch_and_invalid_counts_raise_atomically_then_can_invalidate(self):
        lesson = self.grown(board_cells=10)
        lesson.set_crater(CraterPlan(2, 2, 1, 10, 5))
        for remaining in (0, 4, 6, 10, 11, -1, True, 5.0, "5", None, float("nan")):
            self.assert_atomic(lesson, lambda value=remaining: lesson.apply_damage(value))
        result = lesson.invalidate("Applied cut did not match its preview.")
        self.assertEqual(result["phase"], "invalid")
        self.assertFalse(result["valid"])
        self.assertIsNone(result["remaining_after_cut"])
        self.assertIsNone(result["current"])
        self.assertEqual(result["baseline"], 10)

    def test_no_safe_cut_and_all_dead_growth_are_unavailable(self):
        for baseline in (0, 1, 10):
            lesson = self.grown(baseline=baseline)
            state = lesson.set_crater(None)
            self.assertEqual(state["phase"], "unavailable")
            self.assertTrue(state["valid"])
            self.assertTrue(state["finished"])
            self.assertIn("No safe", state["reason"])
            self.assertIsNone(state["plan"])
            self.assertIsNone(state["removed"])
            self.assertEqual(lesson.clip_steps(8), 0)
            self.assert_atomic(lesson, lambda: lesson.record_tick(3, 100))

    def test_exact_threshold_rounds_display_up_and_resets_hold_after_any_miss(self):
        lesson = self.recovering(LessonConfig(1, 8, 9, 10, 3), baseline=11)
        self.assertEqual(lesson.snapshot()["target"], 10)
        states = [lesson.record_tick(tick, living) for tick, living in
                  enumerate((10, 10, 9, 10, 11, 10), start=2)]
        self.assertEqual([state["hold"] for state in states], [1, 2, 0, 1, 2, 3])
        self.assertEqual(states[-1]["phase"], "complete")
        self.assertEqual(states[-1]["recovery_ticks"], 6)
        exact = self.recovering(LessonConfig(1, 2, 9, 10, 2))
        self.assertEqual(exact.record_tick(2, 9)["hold"], 1)
        self.assertEqual(exact.record_tick(3, 9)["phase"], "complete")

    def test_target_arithmetic_stays_exact_beyond_float_precision(self):
        baseline = 10**400 + 3
        removed = baseline // 2
        lesson = self.recovering(LessonConfig(1, 3, 9, 10, 1), baseline, removed)
        target = (9 * baseline + 9) // 10
        self.assertEqual(lesson.snapshot()["target"], target)
        self.assertEqual(lesson.record_tick(2, target - 1)["hold"], 0)
        self.assertEqual(lesson.record_tick(3, target)["phase"], "complete")
        json.dumps(lesson.result, allow_nan=False)

    def test_complete_at_exact_endpoint_takes_precedence_over_timeout(self):
        lesson = self.recovering(LessonConfig(1, 4, 9, 10, 2))
        for tick, living in enumerate((1, 8, 9, 9), start=2):
            state = lesson.record_tick(tick, living)
        self.assertEqual(state["phase"], "complete")
        self.assertEqual(state["recovery_remaining_ticks"], 0)
        self.assertIsNone(state["reason"])

    def test_horizon_timeout_including_zero_living_never_claims_completion(self):
        for samples in ((0, 0, 0, 0), (9, 8, 9, 8), (8, 8, 8, 9)):
            lesson = self.recovering(LessonConfig(1, 4, 9, 10, 2))
            for tick, living in enumerate(samples, start=2):
                state = lesson.record_tick(tick, living)
            self.assertEqual(state["phase"], "timeout")
            self.assertEqual(state["recovery_ticks"], 4)
            self.assertTrue(state["valid"])
            self.assertIn("not held", state["reason"])
            self.assertTrue(lesson.finished)

    def test_skipped_duplicate_backwards_and_noninteger_ticks_are_atomic(self):
        lesson = Lesson()
        lesson.start_seed()
        for tick in (0, 2, 160, -1, True, np.bool_(True), 1.0, "1", None, float("nan"), float("inf")):
            self.assert_atomic(lesson, lambda value=tick: lesson.record_tick(value, 10))
        lesson.record_tick(1, 10)
        for tick in (0, 1, 3, 4):
            self.assert_atomic(lesson, lambda value=tick: lesson.record_tick(value, 10))
        lesson = self.recovering()
        for tick in (0, 1, 2, 4, 160):
            self.assert_atomic(lesson, lambda value=tick: lesson.record_tick(value, 9))
        self.assertEqual(lesson.record_tick(3, 9)["recovery_ticks"], 1)
        self.assert_atomic(lesson, lambda: lesson.record_tick(3, 9))

    def test_record_count_capacity_and_finite_argument_validation_are_atomic(self):
        lesson = self.recovering(board_cells=10)
        for living in (-1, True, np.bool_(True), 1.0, "1", None, float("nan"), float("inf"), 11):
            self.assert_atomic(lesson, lambda value=living: lesson.record_tick(3, value))
        for finite in (None, 0, 1, "true", np.bool_(True), float("nan")):
            self.assert_atomic(lesson, lambda value=finite: lesson.record_tick(3, 9, finite=value))
        state = lesson.record_tick(np.int64(3), np.int64(9))
        self.assertIs(type(state["tick"]), int)
        self.assertIs(type(state["current"]), int)

    def test_nonfinite_growth_or_recovery_tick_invalidates_and_ignores_counts(self):
        for phase, invalid_tick in (("grow", 1), ("grow", 2), ("recover", 3), ("recover", 7)):
            lesson = Lesson(LessonConfig(2, 5, 9, 10, 2))
            if phase == "grow":
                lesson.start_seed()
            else:
                lesson = self.recovering()
            while lesson.tick + 1 < invalid_tick:
                lesson.record_tick(lesson.tick + 1, 0)
            before = lesson.snapshot()
            state = lesson.record_tick(invalid_tick, object(), finite=False)
            self.assertEqual((state["phase"], state["tick"]), ("invalid", invalid_tick))
            self.assertFalse(state["valid"])
            self.assertTrue(state["finished"])
            self.assertIsNone(state["current"])
            self.assertEqual(state["baseline"], before["baseline"])
            self.assertEqual(state["hold"], before["hold"])
            self.assertIn("Non-finite", state["reason"])
            self.assertEqual(lesson.clip_steps(8), 0)
            self.assert_atomic(lesson, lambda: lesson.record_tick(invalid_tick + 1, 9))

    def test_invalidate_reason_validation_and_gate_invalidation(self):
        lesson = Lesson()
        for reason in (None, 1, True, "", "   ", []):
            self.assert_atomic(lesson, lambda value=reason: lesson.invalidate(value))
        state = lesson.invalidate("The seed could not be verified.")
        self.assertEqual((state["phase"], state["tick"]), ("invalid", 0))
        self.assertEqual(state["reason"], "The seed could not be verified.")
        self.assert_atomic(lesson, lambda: lesson.invalidate("Replacement reason"))

    def test_all_terminal_states_are_frozen_for_every_mutator_until_reset(self):
        for ending in ("complete", "timeout", "unavailable", "invalid"):
            lesson = self.grown(LessonConfig(1, 1, 9, 10, 1))
            if ending == "unavailable":
                lesson.set_crater(None)
            elif ending == "invalid":
                lesson.invalidate("Invalid observation")
            else:
                lesson.set_crater(CraterPlan(2, 2, 1, 10, 5))
                lesson.apply_damage(5)
                lesson.watch()
                lesson.record_tick(2, 9 if ending == "complete" else 0)
            self.assertEqual(lesson.phase, ending)
            for operation in (lesson.start_seed, lesson.watch, lambda: lesson.set_crater(None),
                              lambda: lesson.apply_damage(5), lambda: lesson.record_tick(lesson.tick + 1, 9),
                              lambda: lesson.record_tick(lesson.tick + 1, None, finite=False),
                              lambda: lesson.invalidate("New reason")):
                self.assert_atomic(lesson, operation)
            self.assertEqual(lesson.clip_steps(1000), 0)
            self.assertEqual(lesson.reset()["phase"], "seed")
            self.assertFalse(lesson.finished)

    def test_reset_is_available_in_every_stage_and_does_not_change_old_exports(self):
        lesson = Lesson(LessonConfig(1, 2, 9, 10, 1), board_cells=10)
        initial, config = lesson.snapshot(), lesson.config
        for stage in ("seed", "grow", "damage", "preview", "injured", "recover", "complete"):
            lesson.reset()
            if stage != "seed":
                lesson.start_seed()
            if stage not in ("seed", "grow"):
                lesson.record_tick(1, 10)
            if stage in ("preview", "injured", "recover", "complete"):
                lesson.set_crater(CraterPlan(2, 2, 1, 10, 5))
            if stage in ("injured", "recover", "complete"):
                lesson.apply_damage(5)
            if stage in ("recover", "complete"):
                lesson.watch()
            if stage == "complete":
                lesson.record_tick(2, 9)
            before = lesson.snapshot()
            encoded = json.dumps(before, sort_keys=True)
            self.assertEqual(lesson.reset(), initial)
            self.assertIs(lesson.config, config)
            self.assertEqual(lesson.board_cells, 10)
            self.assertEqual(json.dumps(before, sort_keys=True), encoded)

    def test_snapshot_isolation_json_safety_and_observation_has_no_rng_effect(self):
        lesson = self.recovering()
        before = lesson.snapshot()
        state = lesson.snapshot()
        state["config"]["grow_ticks"] = 1000
        state["plan"]["radius"] = 1000
        state["current"] = 1000
        self.assertEqual(lesson.snapshot(), before)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        for _ in range(5):
            state = lesson.snapshot()
            self.assertEqual(json.loads(json.dumps(state, allow_nan=False)), state)
            self.assertEqual(lesson.clip_steps(8), 5)
        lesson.record_tick(3, 9)
        terminal = lesson.record_tick(4, 9)
        terminal["plan"]["remaining"] = 0
        terminal["config"]["target_den"] = 1
        self.assertEqual(lesson.result["plan"]["remaining"], 5)
        self.assertEqual(lesson.result["config"]["target_den"], 10)
        self.assertEqual(random.getstate(), python_rng)
        after = np.random.get_state()
        np.testing.assert_array_equal(after[1], numpy_rng[1])
        self.assertEqual((after[0], *after[2:]), (numpy_rng[0], *numpy_rng[2:]))

    def test_clip_steps_respects_all_gates_and_caller_stops_batch_on_early_completion(self):
        results = []
        for speed in (1, 2, 4, 8, 1000):
            lesson = Lesson(LessonConfig(5, 9, 9, 10, 2))
            self.assertEqual(lesson.clip_steps(speed), 0)
            lesson.start_seed()
            ticks = []
            while lesson.can_step:
                for _ in range(lesson.clip_steps(speed)):
                    ticks.append(lesson.tick + 1)
                    lesson.record_tick(lesson.tick + 1, 10)
            self.assertEqual(ticks, list(range(1, 6)))
            self.assertEqual(lesson.phase, "damage")
            self.assertEqual(lesson.clip_steps(speed), 0)
            lesson.set_crater(CraterPlan(2, 2, 1, 10, 5))
            self.assertEqual(lesson.clip_steps(speed), 0)
            lesson.apply_damage(5)
            self.assertEqual(lesson.clip_steps(speed), 0)
            lesson.watch()
            while lesson.can_step:
                for _ in range(lesson.clip_steps(speed)):
                    ticks.append(lesson.tick + 1)
                    lesson.record_tick(lesson.tick + 1, 9 if lesson.tick >= 7 else 8)
                    if not lesson.can_step:
                        break
            self.assertEqual(ticks, list(range(1, 10)))
            self.assertEqual(lesson.phase, "complete")
            self.assertEqual(lesson.clip_steps(speed), 0)
            results.append(lesson.result)
        self.assertTrue(all(result == results[0] for result in results))

    def test_clip_steps_validation_is_atomic_even_at_paused_and_terminal_gates(self):
        lesson = Lesson()
        for phase in ("seed", "grow", "invalid"):
            if phase == "grow":
                lesson.start_seed()
            if phase == "invalid":
                lesson.invalidate("Stopped observation")
            self.assertEqual(lesson.clip_steps(0), 0)
            for requested in (-1, True, 1.0, "8", None, float("nan"), float("inf")):
                self.assert_atomic(lesson, lambda value=requested: lesson.clip_steps(value))

    def test_import_has_no_torch_or_pygame_dependency(self):
        source = Path(__file__).resolve().parents[1]
        code = (f"import sys; sys.path.insert(0, {str(source)!r}); import lesson; "
                "assert not {'torch', 'pygame'} & sys.modules.keys(); "
                "assert lesson.Lesson().snapshot()['phase'] == 'seed'")
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
