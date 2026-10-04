import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import duel_recipe as recipes
from duel import DuelConfig, DuelRound


def runtime():
    return {"device": "cpu", "cpu_threads": 1, "deterministic_algorithms": True,
            "python_version": "3.12.0", "torch_version": "2.8.0+cpu", "numpy_version": "2.2.0",
            "platform": "Linux", "machine": "x86_64"}


def completed_report(mode="hard"):
    duel = DuelRound(DuelConfig(4, 1), mode=mode, board_cells=16 ** 2)
    for tick in range(1, 5):
        duel.record_tick(tick, 10 + tick, 20 + tick)
    return {"finite": True, "device": "cpu", "cpu_threads": 1, "step": 4, "seed": 7,
            "mode": mode, "grid_size": 16, "placement": "mirrored", "lesson": None,
            "starting_positions": {"left": [5, 7], "right": [10, 7]},
            "left": {"name": "01_heart", "seed_dir": "seed_003", "health": "ready",
                     "source": "/private/weights/best.pt", "score": .01},
            "right": {"name": "02_star", "seed_dir": "seed_008", "health": "ready"},
            "duel": duel.snapshot(), "elapsed_seconds": 100,
            "rules": {"pressure_gain": .28, "control_decay": .97, "capture_threshold": .55,
                      "release_threshold": .12, "tie_margin": .01}}


def recipe(mode="hard"):
    return recipes.build_recipe(completed_report(mode), runtime())


class RecipeTests(unittest.TestCase):
    def test_fresh_sanitized_recipe_uses_only_live_report(self):
        report, info = completed_report(), {**runtime(), "private": "/secret", "interop_threads": 8}
        before = copy.deepcopy((report, info))
        captured = recipes.build_recipe(report, info)
        self.assertEqual(captured["models"], {
            "left": {"target": "01_heart", "checkpoint_seed": 3},
            "right": {"target": "02_star", "checkpoint_seed": 8}})
        self.assertEqual(captured["expected_score"], {"left_cell_ticks": 39, "right_cell_ticks": 69})
        text = json.dumps(captured, allow_nan=False)
        for forbidden in ("/private", "/secret", "source", "seed_dir", "elapsed", "winner", "averages", "metric", "hash"):
            self.assertNotIn(forbidden, text)
        captured["configuration"]["starting_positions"]["left"][0] = 3
        captured["models"]["left"]["target"] = "changed"
        self.assertEqual((report, info), before)
        self.assertEqual(recipes.build_recipe(report, info), recipe())

    def test_validate_returns_independent_nested_objects(self):
        raw = recipe()
        fresh = recipes.validate_recipe(raw)
        self.assertEqual(fresh, raw)
        fresh["models"]["left"]["checkpoint_seed"] = 9
        fresh["configuration"]["rules"]["pressure_gain"] = 3
        fresh["configuration"]["starting_positions"]["left"][0] = 3
        fresh["runtime"]["python_version"] = "9"
        fresh["expected_score"]["left_cell_ticks"] = 1
        self.assertEqual(raw, recipe())

    def test_import_is_pure_without_numpy_torch_or_pygame(self):
        code = (f"import sys; sys.path.insert(0, {str(V2_ROOT)!r}); import duel_recipe; "
                "assert not {'numpy', 'torch', 'pygame'} & sys.modules.keys()")
        result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_export_requires_completed_finite_ready_cpu_duel(self):
        cases = [("finite", False), ("finite", 1), ("device", "cuda"), ("duel", None),
                 ("lesson", {"phase": "complete"}), ("step", 3), ("step", True), ("cpu_threads", 2)]
        for key, value in cases:
            raw = completed_report()
            raw[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(recipes.RecipeValidationError):
                recipes.build_recipe(raw, runtime())
        for key, value in (("finished", False), ("valid", False), ("phase", "invalid"),
                           ("invalid_reason", "bad"), ("tick", 3), ("scored_ticks", 1), ("mode", "soft")):
            raw = completed_report()
            raw["duel"][key] = value
            with self.subTest(key=key), self.assertRaises(recipes.RecipeValidationError):
                recipes.build_recipe(raw, runtime())
        for key, value in (("health", "collapsed"), ("health", "unverified"), ("seed_dir", "/tmp/seed_003"),
                           ("seed_dir", "seed_4294967296"), ("seed_dir", None)):
            raw = completed_report()
            raw["left"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(recipes.RecipeValidationError):
                recipes.build_recipe(raw, runtime())

    def test_unknown_and_missing_keys_rejected_at_every_schema_object(self):
        paths = [(), ("models",), ("models", "left"), ("models", "right"), ("configuration",),
                 ("configuration", "starting_positions"), ("configuration", "rules"), ("runtime",), ("expected_score",)]
        for path in paths:
            for action in ("add", "remove"):
                raw = recipe()
                obj = raw
                for key in path:
                    obj = obj[key]
                if action == "add":
                    obj["arbitrary"] = "value"
                else:
                    obj.pop(next(iter(obj)))
                with self.subTest(path=path, action=action), self.assertRaises(recipes.RecipeValidationError):
                    recipes.validate_recipe(raw)

    def test_supported_versions_and_bounded_integer_fields(self):
        cases = [((), "schema", "other"), ((), "schema_version", True), ((), "schema_version", 2),
                 ((), "ruleset_version", 2), (("configuration",), "grid_size", 7),
                 (("configuration",), "grid_size", 129), (("configuration",), "round_ticks", 0),
                 (("configuration",), "round_ticks", 10001), (("configuration",), "warmup_ticks", 4),
                 (("configuration",), "warmup_ticks", -1), (("configuration",), "simulation_seed", -1),
                 (("configuration",), "simulation_seed", 2**32), (("runtime",), "cpu_threads", 0),
                 (("runtime",), "cpu_threads", 65), (("models", "left"), "checkpoint_seed", -1),
                 (("models", "left"), "checkpoint_seed", 2**32)]
        for path, key, value in cases:
            raw = recipe()
            obj = raw
            for part in path:
                obj = obj[part]
            obj[key] = value
            with self.subTest(path=path, key=key, value=value), self.assertRaises(recipes.RecipeValidationError):
                recipes.validate_recipe(raw)
        integer_paths = [((), "schema_version"), (("configuration",), "grid_size"),
                         (("configuration",), "simulation_seed"), (("configuration",), "round_ticks"),
                         (("configuration",), "warmup_ticks"), (("models", "left"), "checkpoint_seed"),
                         (("runtime",), "cpu_threads"), (("expected_score",), "left_cell_ticks")]
        for path, key in integer_paths:
            for value in (True, False, 1.0, "1", None):
                raw = recipe()
                obj = raw
                for part in path:
                    obj = obj[part]
                obj[key] = value
                with self.subTest(path=path, key=key, value=value), self.assertRaises(recipes.RecipeValidationError):
                    recipes.validate_recipe(raw)

    def test_boundary_values_work_without_clamping(self):
        for size in (8, 128):
            raw = recipe()
            config = raw["configuration"]
            config.update(grid_size=size, placement="custom", round_ticks=10000, warmup_ticks=9999,
                          simulation_seed=2**32 - 1, starting_positions={"left": [2, 2], "right": [size - 3, size - 3]})
            raw["models"]["left"]["checkpoint_seed"] = 2**32 - 1
            raw["runtime"]["cpu_threads"] = 64
            raw["expected_score"] = {"left_cell_ticks": size ** 2, "right_cell_ticks": 0}
            self.assertEqual(recipes.validate_recipe(raw), raw)
        raw = recipe()
        raw["configuration"].update(round_ticks=1, warmup_ticks=0)
        self.assertEqual(recipes.validate_recipe(raw), raw)

    def test_positions_match_reset_world_and_mirrored_convention(self):
        for pair in ([1, 7], [14, 7], [5, 1], [5, 14], [True, 7], [5.0, 7], [5], (5, 7), None):
            raw = recipe()
            raw["configuration"]["placement"] = "custom"
            raw["configuration"]["starting_positions"]["left"] = pair
            with self.subTest(pair=pair), self.assertRaises(recipes.RecipeValidationError):
                recipes.validate_recipe(raw)
        raw = recipe()
        raw["configuration"]["starting_positions"]["left"] = [2, 2]
        with self.assertRaisesRegex(recipes.RecipeValidationError, "mirrored"):
            recipes.validate_recipe(raw)
        raw["configuration"]["placement"] = "custom"
        raw["configuration"]["starting_positions"]["right"] = [2, 2]
        self.assertEqual(recipes.validate_recipe(raw), raw)  # Same legal cell is an explicit custom start.

    def test_score_feasibility_hard_joint_bound_and_soft_overlap(self):
        for key, value in (("left_cell_ticks", -1), ("left_cell_ticks", 769), ("right_cell_ticks", True)):
            raw = recipe()
            raw["expected_score"][key] = value
            with self.assertRaises(recipes.RecipeValidationError):
                recipes.validate_recipe(raw)
        raw = recipe()
        raw["expected_score"] = {"left_cell_ticks": 768, "right_cell_ticks": 1}
        with self.assertRaises(recipes.RecipeValidationError):
            recipes.validate_recipe(raw)
        raw["configuration"]["mode"] = "soft"
        raw["expected_score"]["right_cell_ticks"] = 768
        self.assertEqual(recipes.validate_recipe(raw), raw)

    def test_rule_bounds_finiteness_and_no_numeric_bool_coercion(self):
        for key in recipes.RULE_KEYS:
            for value in (True, False, float("nan"), float("inf"), -float("inf"), "0.5", 10**400):
                raw = recipe()
                raw["configuration"]["rules"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(recipes.RecipeValidationError):
                    recipes.validate_recipe(raw)
        for key, value in (("pressure_gain", 0), ("control_decay", 1.1), ("tie_margin", -1),
                           ("capture_threshold", .1), ("release_threshold", .55)):
            raw = recipe()
            raw["configuration"]["rules"][key] = value
            with self.assertRaises(recipes.RecipeValidationError):
                recipes.validate_recipe(raw)

    def test_targets_and_runtime_diagnostics_are_bounded_non_path_strings(self):
        for target in ("../01_heart", "/tmp/01_heart", "01_heart.png", "C:\\01_heart", "", "a" * 81, "x\n"):
            raw = recipe()
            raw["models"]["left"]["target"] = target
            with self.assertRaises(recipes.RecipeValidationError):
                recipes.validate_recipe(raw)
        for key in recipes.DIAGNOSTIC_KEYS:
            for value in ("", "x" * 129, "/private/path", "C:\\private", "x\x1b[2J", "x\n", 123):
                raw = recipe()
                raw["runtime"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(recipes.RecipeValidationError):
                    recipes.validate_recipe(raw)
        for value in (1, "true", None):
            raw = recipe()
            raw["runtime"]["deterministic_algorithms"] = value
            with self.assertRaises(recipes.RecipeValidationError):
                recipes.validate_recipe(raw)
        raw = recipe()
        raw["runtime"]["deterministic_algorithms"] = False
        self.assertEqual(recipes.validate_recipe(raw), raw)

    def test_json_duplicate_nonfinite_malformed_utf8_and_size_limit(self):
        base = json.dumps(recipe())
        cases = [base.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'),
                 base.replace('"target": "01_heart"', '"target": "01_heart", "target": "01_heart"'),
                 base.replace('"pressure_gain": 0.28', '"pressure_gain": NaN'),
                 base.replace('"pressure_gain": 0.28', '"pressure_gain": Infinity'),
                 base.replace('"pressure_gain": 0.28', '"pressure_gain": 1e999'),
                 "[]", "null", "{bad}", "[" * 2000, " " * (recipes.MAX_FILE_BYTES + 1)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            for value in cases:
                path.write_text(value)
                with self.subTest(value=value[:100]), self.assertRaises(recipes.RecipeValidationError):
                    recipes.load_recipe(path)
            path.write_bytes(b"\xff")
            with self.assertRaises(recipes.RecipeValidationError):
                recipes.load_recipe(path)
            path.write_text(base + " " * (recipes.MAX_FILE_BYTES - len(base)))
            self.assertEqual(recipes.load_recipe(path), recipe())

    def test_atomic_replace_preserves_existing_output_on_all_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duel.json"
            path.write_bytes(b"keep me")
            invalid = recipe()
            invalid["runtime"]["cpu_threads"] = 100
            with self.assertRaises(recipes.RecipeValidationError):
                recipes.write_recipe(path, invalid)
            for target in ("duel_recipe.json.dumps", "duel_recipe.os.replace", "duel_recipe.os.fsync"):
                with patch(target, side_effect=OSError("failed")), self.assertRaises(OSError):
                    recipes.write_recipe(path, recipe())
                self.assertEqual(path.read_bytes(), b"keep me")
                self.assertEqual(list(Path(folder).iterdir()), [path])
            self.assertEqual(recipes.write_recipe(path, recipe()), path)
            self.assertEqual(recipes.load_recipe(path), recipe())

    def test_unique_saves_are_race_safe_and_do_not_consume_global_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            saved_rng = random.getstate()
            with ThreadPoolExecutor(max_workers=8) as executor:
                paths = list(executor.map(lambda _: recipes.save_unique_recipe(folder, recipe()), range(24)))
            self.assertEqual(len(set(paths)), 24)
            self.assertEqual(random.getstate(), saved_rng)
            self.assertTrue(all(recipes.load_recipe(path) == recipe() for path in paths))
            self.assertEqual(len(list(Path(folder).iterdir())), 24)
            with patch("duel_recipe.os.fsync", side_effect=OSError("failed")), self.assertRaises(OSError):
                recipes.save_unique_recipe(folder, recipe())
            self.assertEqual(len(list(Path(folder).iterdir())), 24)

    def test_path_aliases_cover_resolved_paths_symlinks_and_hardlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            path, different = Path(folder) / "a", Path(folder) / "different"
            path.touch()
            hard, soft = Path(folder) / "hard", Path(folder) / "soft"
            os.link(path, hard)
            soft.symlink_to(path)
            for alias in (path, path.parent / "." / "a", hard, soft):
                self.assertTrue(recipes.paths_alias(path, alias))
            self.assertFalse(recipes.paths_alias(path, different))


if __name__ == "__main__":
    unittest.main()
