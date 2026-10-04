import contextlib
import copy
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import arena
import clash
import duel_recipe
import replay
from runtime import runtime_info


class GrowingModel(torch.nn.Module):
    def __init__(self, probability=.5):
        super().__init__()
        self.probability = probability

    def forward(self, state, steps=1):
        result = state.clone()
        alpha = result[:, 3:4]
        growth = (F.max_pool2d(alpha, 3, stride=1, padding=1) > .1) & (torch.rand_like(alpha) < self.probability)
        result[:, 3:4] = torch.where(growth, 1., alpha)
        return result


def synthetic_bundle(target, device, bootstrap_steps=0, preferred_seed=None, allow_unhealthy=False):
    assert bootstrap_steps == 0 and allow_unhealthy is False and device == "cpu"
    return {"model": GrowingModel(.35 if target.stem == "01_heart" else .6),
            "channels": 8, "grid_size": 16, "channels_last": False, "name": target.stem,
            "seed_dir": f"seed_{preferred_seed if preferred_seed is not None else 0:03d}", "health": "ready",
            "source": "/private/absolute/checkpoint.pt", "score": .01, "kind": "v2"}


def finished_world(*extra):
    args = arena.parse_args(["--duel", "--device", "cpu", "--cpu-threads", "1", "--grid-size", "16",
                             "--round-ticks", "12", "--warmup-ticks", "2", "--seed", "7",
                             "--left-seed", "3", "--right-seed", "8", *extra])
    world = arena.Arena(args, clash.list_targets())
    world.step(args.round_ticks)
    return world


def capture(world):
    report = world.report(0)
    report["finite"] = all(torch.isfinite(value).all().item() for value in (world.a, world.b, world.owner, world.control))
    return duel_recipe.build_recipe(report, runtime_info())


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.threads, self.deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
        self.python_rng, self.numpy_rng, self.torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.loader = patch("arena.ensure_model", side_effect=synthetic_bundle)
        self.load = self.loader.start()

    def tearDown(self):
        self.loader.stop()
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(self.deterministic)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)

    def test_complete_synthetic_hard_and_soft_rounds_match_scores_and_state(self):
        for mode in ("hard", "soft"):
            original = finished_world("--mode", mode)
            saved = capture(original)
            before = copy.deepcopy(saved)
            report, world = replay.run_replay(saved)
            self.assertEqual(report["status"], "matched")
            self.assertEqual(report["expected_score"], report["observed_score"])
            self.assertEqual(world.steps, 12)
            self.assertEqual(world.duel.snapshot(), original.duel.snapshot())
            for key in ("a", "b", "owner", "control"):
                self.assertTrue(torch.equal(getattr(original, key), getattr(world, key)), key)
            self.assertEqual(saved, before)
            text = json.dumps(report, allow_nan=False)
            for private in ("/private/", "source", "seed_dir", "elapsed", "fingerprint"):
                self.assertNotIn(private, text)

    def test_changed_live_cultures_mode_and_custom_rules_are_captured(self):
        world = finished_world()
        world.select(5, 0)
        world.select(6, 1)
        world.mode = "soft"
        world.args.left_pos, world.args.right_pos = (2, 3), (13, 12)
        world.args.pressure_gain = .4
        world.reset()
        world.step(12)
        self.assertEqual((world.args.left, world.args.right, world.args.mode), (1, 2, "hard"))
        saved = capture(world)
        self.assertEqual(saved["models"]["left"]["target"], "06_flower")
        self.assertEqual(saved["models"]["right"]["target"], "07_umbrella")
        self.assertEqual(saved["configuration"]["mode"], "soft")
        report, observed = replay.run_replay(saved)
        self.assertEqual(report["status"], "matched")
        self.assertEqual(observed.start_positions, {"left": [2, 3], "right": [13, 12]})
        self.assertEqual(observed.args.pressure_gain, .4)

    def test_export_and_unique_save_do_not_mutate_arena_or_consume_simulation_rng(self):
        world = finished_world()
        states = [getattr(world, key).clone() for key in ("a", "b", "owner", "control")]
        result, args = world.duel.snapshot(), copy.deepcopy(vars(world.args))
        py, np_state, pt = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with tempfile.TemporaryDirectory() as folder:
            saved = capture(world)
            duel_recipe.save_unique_recipe(folder, saved)
            duel_recipe.write_recipe(Path(folder) / "explicit.json", saved)
        self.assertEqual(random.getstate(), py)
        observed_np = np.random.get_state()
        self.assertEqual(observed_np[0], np_state[0])
        self.assertTrue(np.array_equal(observed_np[1], np_state[1]))
        self.assertEqual(observed_np[2:], np_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), pt))
        self.assertEqual(world.duel.snapshot(), result)
        self.assertEqual(vars(world.args), args)
        for key, expected in zip(("a", "b", "owner", "control"), states):
            self.assertTrue(torch.equal(getattr(world, key), expected))

    def test_exact_seed_no_bootstrap_or_model_override_and_catalog_reordering(self):
        saved = capture(finished_world())
        self.load.reset_mock()
        catalog = list(reversed(clash.list_targets()))
        args = replay.recipe_to_args(saved, catalog)
        self.assertEqual((args.device, args.bootstrap_steps, args.allow_unhealthy), ("cpu", 0, False))
        self.assertEqual((args.left, args.right), (9, 8))
        self.assertEqual((args.left_seed, args.right_seed), (3, 8))
        report, _ = replay.run_replay(saved, catalog)
        self.assertEqual(report["status"], "matched")
        self.assertEqual([(call.args[0].stem, call.kwargs["preferred_seed"]) for call in self.load.call_args_list],
                         [("01_heart", 3), ("02_star", 8)])
        with self.assertRaises(replay.ReplayLoadError):
            replay.resolve_recipe_models(saved, catalog[1:8])
        with self.assertRaises(replay.ReplayLoadError):
            replay.resolve_recipe_models(saved, [*catalog, catalog[0]])

    def test_one_cell_tick_difference_is_not_a_score_match(self):
        saved = capture(finished_world())
        saved["expected_score"]["left_cell_ticks"] += 1
        report, _ = replay.run_replay(saved)
        self.assertEqual(report["status"], "different")
        self.assertEqual(report["expected_score"]["left_cell_ticks"] - report["observed_score"]["left_cell_ticks"], 1)
        self.assertIn("does not verify checkpoint identity", report["interpretation"])

    def test_runtime_version_differences_warn_but_still_check_scores(self):
        saved = capture(finished_world())
        for key in duel_recipe.DIAGNOSTIC_KEYS:
            saved["runtime"][key] = "other"
        report, _ = replay.run_replay(saved)
        self.assertEqual(report["status"], "matched")
        self.assertEqual(len(report["warnings"]), len(duel_recipe.DIAGNOSTIC_KEYS))
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            replay.print_summary(report)
        self.assertIn("Warning: Runtime differs", err.getvalue())
        self.assertIn("Expected:", out.getvalue())
        self.assertIn("Observed:", out.getvalue())

    def test_runtime_warnings_arrive_before_step_and_main_does_not_repeat(self):
        saved = capture(finished_world())
        saved["runtime"]["python_version"] = "other"
        events = []
        original_step = arena.Arena.step
        def step(world, count):
            events.append("step")
            return original_step(world, count)
        with patch.object(arena.Arena, "step", step):
            report, _ = replay.run_replay(saved, on_warning=lambda warning: events.append("warning"))
        self.assertEqual(events, ["warning", "step"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recipe.json"
            duel_recipe.write_recipe(path, saved)
            with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(replay.main([str(path)]), 0)
            self.assertEqual(err.getvalue().count("Runtime differs"), 1)

    def test_loader_error_text_never_leaks_paths_into_cli_diagnostics(self):
        saved = capture(finished_world())
        self.load.side_effect = ValueError("bad checkpoint at /private/secret/best.pt")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recipe.json"
            duel_recipe.write_recipe(path, saved)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(replay.main([str(path)]), 2)
            self.assertNotIn("/private", err.getvalue())
            self.assertIn("ValueError", err.getvalue())
            self.assertIn("01_heart [seed_3]", err.getvalue())
            self.assertIn("no substitutes", err.getvalue())

    def test_missing_exact_seed_never_falls_back_to_another_ready_seed(self):
        saved = capture(finished_world())
        self.loader.stop()
        with tempfile.TemporaryDirectory() as folder, patch.object(clash, "V2_ROOT", Path(folder)):
            other = Path(folder) / "weights" / "01_heart" / "seed_000"
            (other / "checkpoints").mkdir(parents=True)
            (other / "checkpoints" / "best.pt").touch()
            (other / "best_summary.json").write_text('{"score": 0.01}')
            with patch.object(clash, "load_v2_model") as load, self.assertRaises(replay.ReplayLoadError):
                replay.run_replay(saved, [Path("01_heart.png"), Path("02_star.png")])
            load.assert_not_called()

    def test_snapshot_failure_preserves_existing_image(self):
        world = finished_world()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "board.png"
            path.write_bytes(b"keep image")
            with patch("PIL.Image.Image.save", side_effect=OSError("encoder failed")), self.assertRaises(OSError):
                replay.write_snapshot(path, world)
            self.assertEqual(path.read_bytes(), b"keep image")
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_recorded_cpu_threads_ignore_environment_and_determinism_is_applied(self):
        saved = capture(finished_world())
        saved["runtime"]["deterministic_algorithms"] = False
        with patch.dict(os.environ, {"PETRI_CPU_THREADS": "999"}):
            report, _ = replay.run_replay(saved)
        self.assertEqual(report["runtime"]["cpu_threads"], 1)
        self.assertFalse(report["runtime"]["deterministic_algorithms"])
        self.assertEqual(report["status"], "matched")

    def test_loaded_identity_and_health_drift_refused_before_any_simulation(self):
        saved = capture(finished_world())
        for key, value in (("name", "02_star"), ("seed_dir", "seed_000"), ("health", "collapsed"), ("seed_dir", "/secret/seed_003")):
            def bad(*args, **kwargs):
                bundle = synthetic_bundle(*args, **kwargs)
                bundle[key] = value
                return bundle
            self.load.side_effect = bad
            with patch.object(arena.Arena, "step") as step, self.subTest(key=key), self.assertRaises(replay.ReplayLoadError):
                replay.run_replay(saved)
            step.assert_not_called()

    def test_nonfinite_and_failed_simulations_never_award_match(self):
        saved = capture(finished_world())
        class Nonfinite(torch.nn.Module):
            def forward(self, value, steps=1):
                return torch.full_like(value, float("nan"))
        def bad(*args, **kwargs):
            result = synthetic_bundle(*args, **kwargs)
            result["model"] = Nonfinite()
            return result
        self.load.side_effect = bad
        for mode in ("hard", "soft"):
            saved["configuration"]["mode"] = mode
            report, world = replay.run_replay(saved)
            self.assertEqual(report["status"], "invalid")
            self.assertEqual(world.steps, 1)
            self.assertIsNone(report["observed_score"])
            self.assertIsNone(report["result"])
        self.load.side_effect = synthetic_bundle
        with patch.object(arena.Arena, "step", side_effect=RuntimeError("failure /private/path")):
            report, _ = replay.run_replay(saved)
        self.assertEqual(report["error"], {"stage": "simulation", "type": "RuntimeError"})
        self.assertNotIn("/private", json.dumps(report))

    def test_cli_only_accepts_input_and_output_controls(self):
        for extra in (["--mode", "soft"], ["--device", "cuda"], ["--left", "6"], ["--seed", "4"],
                      ["--round-ticks", "5"], ["--checkpoint", "/tmp/file"], ["--allow-unhealthy"],
                      ["--bootstrap-steps", "5"], ["--rep", "foo"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                replay.parse_args(["recipe.json", *extra])

    def test_cli_statuses_json_snapshot_optional_and_existing_outputs_preserved(self):
        saved = capture(finished_world())
        with tempfile.TemporaryDirectory() as folder:
            source, report_path, image = (Path(folder) / name for name in ("recipe.json", "check.json", "board.png"))
            duel_recipe.write_recipe(source, saved)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(replay.main([str(source)]), 0)
            self.assertIn("Saved scores matched", out.getvalue())
            self.assertFalse(report_path.exists())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(replay.main([str(source), "--report", str(report_path), "--snapshot", str(image)]), 0)
            self.assertTrue(image.read_bytes().startswith(b"\x89PNG"))
            from PIL import Image
            with Image.open(image) as opened:
                self.assertEqual(opened.size, (768, 768))
            self.assertEqual(json.loads(report_path.read_text())["status"], "matched")
            saved["expected_score"]["left_cell_ticks"] += 1
            duel_recipe.write_recipe(source, saved)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(replay.main([str(source)]), 1)
            report_path.write_text("keep me")
            for target in ("replay.json.dumps", "replay.os.replace", "replay.os.fsync"):
                with patch(target, side_effect=OSError("output error")), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(replay.main([str(source), "--report", str(report_path)]), 2)
                self.assertEqual(report_path.read_text(), "keep me")
            source.write_text("{invalid}")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(replay.main([str(source)]), 2)
                self.assertEqual(replay.main([str(Path(folder) / "absent.json")]), 2)

    def test_input_output_aliases_fail_before_loading_preserve_every_file(self):
        saved = capture(finished_world())
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "recipe.json"
            duel_recipe.write_recipe(source, saved)
            original = source.read_bytes()
            hard, soft = Path(folder) / "hard.json", Path(folder) / "soft.json"
            os.link(source, hard)
            soft.symlink_to(source)
            for option in ("--report", "--snapshot"):
                for destination in (source, hard, soft):
                    with patch("replay.run_replay") as run, contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(replay.main([str(source), option, str(destination)]), 2)
                    run.assert_not_called()
                    self.assertEqual(source.read_bytes(), original)
            out = Path(folder) / "out"
            with patch("replay.run_replay") as run, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(replay.main([str(source), "--report", str(out), "--snapshot", str(out)]), 2)
            run.assert_not_called()

    def test_missing_collapsed_unverified_and_corrupt_actual_checkpoints_refused(self):
        saved = capture(finished_world())
        self.loader.stop()
        with tempfile.TemporaryDirectory() as folder, patch.object(clash, "V2_ROOT", Path(folder)):
            targets = [Path(folder) / "targets" / f"{name}.png" for name in ("01_heart", "02_star")]
            for target in targets:
                target.parent.mkdir(exist_ok=True)
                target.touch()
            for state in ("missing", "collapsed", "unverified", "corrupt"):
                with self.subTest(state=state):
                    seed = Path(folder) / "weights" / "01_heart" / "seed_003"
                    if state != "missing":
                        (seed / "checkpoints").mkdir(parents=True, exist_ok=True)
                        torch.save({"not_model": "corrupt"}, seed / "checkpoints" / "best.pt")
                        summary = seed / "best_summary.json"
                        if state != "unverified":
                            summary.write_text(json.dumps({"score": .9 if state == "collapsed" else .01}))
                        else:
                            summary.unlink(missing_ok=True)
                    with patch.object(arena.Arena, "step") as step, self.assertRaises(replay.ReplayLoadError):
                        replay.run_replay(saved, targets)
                    step.assert_not_called()

    def test_invalid_simulation_cli_returns_one_and_can_report_without_snapshot(self):
        saved = capture(finished_world())
        with tempfile.TemporaryDirectory() as folder:
            source, report, image = (Path(folder) / name for name in ("recipe.json", "check.json", "board.png"))
            duel_recipe.write_recipe(source, saved)
            with patch.object(arena.Arena, "step", side_effect=RuntimeError("invalid")), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(replay.main([str(source), "--report", str(report), "--snapshot", str(image)]), 1)
            self.assertEqual(json.loads(report.read_text())["status"], "invalid")
            self.assertFalse(image.exists())

    def test_headless_subprocess_never_imports_pygame_initializes_ui_or_sleeps(self):
        code = f'''
import sys, tempfile
from pathlib import Path
sys.path.insert(0, {str(V2_ROOT)!r})
sys.path.insert(0, {str(V2_ROOT / 'tests')!r})
class NoPygame:
    def find_spec(self, fullname, *args):
        if fullname == "pygame" or fullname.startswith("pygame."):
            raise AssertionError("replay imported pygame")
sys.meta_path.insert(0, NoPygame())
from unittest.mock import patch
import arena, duel_recipe, replay
from test_replay import synthetic_bundle, finished_world, capture
arena.ensure_model = synthetic_bundle
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "recipe.json"
    duel_recipe.write_recipe(path, capture(finished_world()))
    with patch("time.sleep", side_effect=AssertionError("slept")), patch("arena.ArenaUI", side_effect=AssertionError("UI")):
        assert replay.main([str(path), "--snapshot", str(Path(folder) / "board.png")]) == 0
assert "pygame" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(os.getenv("PETRI_TEST_CHECKPOINTS") == "1", "opt-in full shipped-model replay")
    def test_shipped_full_round_and_deliberate_one_cell_tick_mismatch(self):
        self.loader.stop()
        original = finished_world("--grid-size", "48", "--round-ticks", "600", "--warmup-ticks", "60",
                                  "--left-seed", "0", "--right-seed", "1")
        saved = capture(original)
        report, _ = replay.run_replay(saved)
        self.assertEqual(report["status"], "matched")
        self.assertEqual(report["result"]["scored_ticks"], 540)
        saved["expected_score"]["left_cell_ticks"] += 1
        report, _ = replay.run_replay(saved)
        self.assertEqual(report["status"], "different")


if __name__ == "__main__":
    unittest.main()
