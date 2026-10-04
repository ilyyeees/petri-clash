"""Exact paired aggregation and small, genuinely headless Arena comparisons."""

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
import compare
from duel import DuelConfig, DuelRound


def scored_leg(seed, swapped, left, right):
    duel = DuelRound(DuelConfig(1, 0))
    return {"seed": seed, "sides": compare.ORIENTATIONS[swapped].copy(), "status": "complete",
            "result": duel.record_tick(1, left, right)}


class AggregationTests(unittest.TestCase):
    def test_swapped_mapping_exact_pair_scores_and_winner_difference(self):
        legs = [scored_leg(7, 0, 11, 3), scored_leg(7, 1, 8, 2)]
        summary = compare.aggregate_pairs([7], legs)
        self.assertEqual(summary["scores"], {"A": 13, "B": 11})
        self.assertEqual(summary["paired_outcome"], "A")
        pair = summary["per_seed"][0]
        self.assertEqual(pair["leg_winners"], ["A", "B"])
        self.assertTrue(pair["winner_changes_after_swap"])

    def test_huge_integers_and_exact_tie_are_never_float_comparisons(self):
        for huge in (2**60, 10**400):
            for margin in (0, 1):
                legs = [scored_leg(0, 0, huge, huge), scored_leg(0, 1, huge, huge + margin)]
                summary = compare.aggregate_pairs([0], legs)
                self.assertEqual(summary["paired_outcome"], "A" if margin else "draw")
                self.assertEqual(summary["scores"]["A"] - summary["scores"]["B"], margin)
                self.assertIs(type(summary["scores"]["A"]), int)
                json.dumps(summary, allow_nan=False)

    def test_empty_draws_and_totals_are_pooled_cell_ticks_not_win_counts(self):
        legs = [scored_leg(0, 0, 0, 0), scored_leg(0, 1, 0, 0),
                scored_leg(7, 0, 11, 0), scored_leg(7, 1, 0, 11),
                scored_leg(42, 0, 0, 1), scored_leg(42, 1, 1, 0)]
        summary = compare.aggregate_pairs([0, 7, 42], legs)
        self.assertEqual(summary["scores"], {"A": 22, "B": 2})
        self.assertEqual(summary["paired_outcome"], "A")
        self.assertEqual([p["paired_outcome"] for p in summary["per_seed"]], ["draw", "A", "B"])

    def test_missing_failed_or_invalid_leg_suppresses_overall_outcome(self):
        completed = [scored_leg(0, 0, 9, 0), scored_leg(0, 1, 0, 9)]
        for extra in ([], [scored_leg(7, 0, 100, 0)]):
            summary = compare.aggregate_pairs([0, 7], completed + extra)
            self.assertEqual(summary["status"], "incomplete")
            self.assertIsNone(summary["scores"])
            self.assertIsNone(summary["paired_outcome"])
            self.assertEqual(summary["completed_pair_scores"], {"A": 18, "B": 0})
        for status in ("invalid", "failed", "complete"):
            bad = scored_leg(7, 0, 100, 0)
            bad["status"] = status
            bad["result"]["valid"] = False
            summary = compare.aggregate_pairs([0, 7], completed + [bad])
            self.assertEqual(summary["status"], "invalid")
            self.assertIsNone(summary["paired_outcome"])

    def test_malformed_terminal_evidence_never_becomes_a_winner(self):
        fields = [("finished", False), ("phase", "scoring"), ("tick", 0), ("tick", True),
                  ("scored_ticks", 1.0), ("winner", "right"), ("mode", "bad"), ("invalid_reason", "bad"),
                  ("scores", {"left": 3.0, "right": 0}),
                  ("scores", {"left": -1, "right": 0}), ("scores", {"left": True, "right": 0})]
        for key, value in fields:
            legs = [scored_leg(0, 0, 3, 0), scored_leg(0, 1, 0, 3)]
            legs[0]["result"][key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(compare.aggregate_pairs([0], legs)["status"], "invalid")

    def test_mismatched_round_settings_are_invalid(self):
        legs = [scored_leg(0, 0, 3, 0), scored_leg(0, 1, 0, 3)]
        duel = DuelRound(DuelConfig(2, 0), mode="soft")
        duel.record_tick(1, 0, 3)
        legs[1]["result"] = duel.record_tick(2, 0, 3)
        summary = compare.aggregate_pairs([0], legs)
        self.assertEqual(summary["status"], "invalid")
        self.assertIsNone(summary["paired_outcome"])

    def test_wrong_seed_mapping_or_duplicate_leg_is_rejected(self):
        first = scored_leg(0, 0, 3, 0)
        for legs in ([first, first], [scored_leg(7, 0, 1, 0)],
                     [{**first, "sides": {"left": "A", "right": "A"}}]):
            with self.assertRaises(ValueError):
                compare.aggregate_pairs([0], legs)

    def test_inputs_and_returned_snapshots_are_independent(self):
        seeds, legs = [0], [scored_leg(0, 0, 9, 0), scored_leg(0, 1, 0, 9)]
        saved = copy.deepcopy(legs)
        first = compare.aggregate_pairs(seeds, legs)
        expected = copy.deepcopy(first)
        first["scores"]["A"] = 999
        first["per_seed"][0]["scores"]["A"] = 999
        first["per_seed"][0]["leg_winners"][0] = "B"
        first["per_seed"][0]["leg_indices"].clear()
        self.assertEqual(legs, saved)
        self.assertEqual(seeds, [0])
        self.assertEqual(compare.aggregate_pairs(seeds, legs), expected)

    def test_pure_aggregation_import_requires_no_torch_numpy_or_pygame(self):
        code = (f"import sys; sys.path.insert(0, {str(V2_ROOT)!r}); import compare; "
                "assert not {'torch', 'numpy', 'pygame'} & sys.modules.keys(); "
                "assert compare.aggregate_pairs([0], [])['paired_outcome'] is None")
        result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class GrowingModel(torch.nn.Module):
    """Stochastic local growth for testing the actual Arena, no saved weights."""
    def __init__(self, probability=.5):
        super().__init__()
        self.probability = probability

    def forward(self, state, steps=1):
        result = state.clone()
        alpha = result[:, 3:4]
        near = F.max_pool2d(alpha, 3, stride=1, padding=1) > .1
        growth = near & (torch.rand_like(alpha) < self.probability)
        result[:, 3:4] = torch.where(growth, 1., alpha)
        return result


def synthetic_bundle(target, device, bootstrap_steps=0, preferred_seed=None, allow_unhealthy=False):
    assert bootstrap_steps == 0 and allow_unhealthy is False
    return {"model": GrowingModel(.35 if target.stem == "01_heart" else .6),
            "channels": 8, "grid_size": 16, "channels_last": False, "name": target.stem,
            "seed_dir": f"seed_{preferred_seed or 0:03d}", "health": "ready",
            "source": "/private/absolute/checkpoint.pt", "score": 0.01, "kind": "v2"}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
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

    def args(self, *extra):
        return compare.parse_args(["--grid-size", "16", "--round-ticks", "12", "--warmup-ticks", "2", *extra])

    def test_defaults_are_six_cpu_rounds_600_60_heart_star(self):
        args = compare.parse_args([])
        self.assertEqual((args.left, args.right, args.seeds), (1, 2, (0, 7, 42)))
        self.assertEqual((args.device, args.round_ticks, args.warmup_ticks), ("cpu", 600, 60))
        self.assertEqual(args.bootstrap_steps, 0)
        self.assertTrue(args.duel)

    def test_seed_validation_valid_extremes_and_different_values(self):
        self.assertEqual(compare.parse_seeds("0, 7,4294967295"), (0, 7, 2**32 - 1))
        self.assertEqual(compare.validate_seeds([0, np.int64(7)]), (0, 7))
        for values in ([], [1, 1], [-1], [2**32], [True], [1.0], ["7"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compare.validate_seeds(values)

    def test_bad_cli_rejected_before_model_loading(self):
        cases = [["--seeds", value] for value in ("", "0,", "0,,7", "7,7", "0,00", "-1", "4294967296", "1.5", "nan", "1_0", "a", "+1")]
        cases += [["--round-ticks", "0"], ["--warmup-ticks", "600"], ["--grid-size", "7"],
                  ["--cpu-threads", "0"], ["--left", "0"], ["--left-seed", "-1"],
                  ["--right-seed", "4294967296"], ["--bootstrap-steps", "5"],
                  ["--checkpoint", "/tmp/checkpoint.pt"], ["--allow-unhealthy"], ["--mode", "unknown"]]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                compare.parse_args(argv)
        with patch.dict(os.environ, {"PETRI_CPU_THREADS": "bad"}), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            compare.parse_args([])
        with self.assertRaises(ValueError):
            compare.run_comparison(self.args("--right", "999"))
        self.load.assert_not_called()

    def test_actual_arena_both_modes_seed_replay_and_no_argument_mutation(self):
        for mode in ("hard", "soft"):
            args = self.args("--mode", mode)
            saved = copy.deepcopy(vars(args))
            first, second = compare.run_comparison(args), compare.run_comparison(args)
            self.assertEqual(first["summary"], second["summary"])
            self.assertEqual([leg["result"] for leg in first["legs"]], [leg["result"] for leg in second["legs"]])
            self.assertEqual(first["summary"]["status"], "complete")
            self.assertEqual(len(first["legs"]), 6)
            self.assertEqual(vars(args), saved)
            self.assertEqual(first["configuration"]["starting_positions"], {"left": [5, 7], "right": [10, 7]})
            self.assertEqual(first["runtime"]["device"], "cpu")
            self.assertIn("torch_version", first["runtime"])
            self.assertTrue(all(leg["result"]["scored_ticks"] == 10 for leg in first["legs"]))
            self.assertGreater(len({tuple(leg["result"]["scores"].values()) for leg in first["legs"]}), 1)
            text = json.dumps(first, allow_nan=False)
            self.assertNotIn("/private/", text)
            self.assertNotIn("fingerprint", text)

    def test_explicit_checkpoint_seeds_follow_culture_in_both_orientations(self):
        report = compare.run_comparison(self.args("--left-seed", "3", "--right-seed", "8", "--seeds", "7"))
        observed = [(call.args[0].stem, call.kwargs["preferred_seed"]) for call in self.load.call_args_list]
        self.assertEqual(observed, [("01_heart", 3), ("02_star", 8), ("02_star", 8), ("01_heart", 3)])
        self.assertEqual(report["models"]["A"]["seed_dir"], "seed_003")
        self.assertEqual(report["models"]["B"]["seed_dir"], "seed_008")
        self.assertEqual(report["legs"][0]["models"], report["legs"][1]["models"])

    def test_same_culture_keeps_distinct_a_b_labels(self):
        report = compare.run_comparison(self.args("--left", "1", "--right", "1", "--seeds", "0"))
        self.assertEqual(set(report["models"]), {"A", "B"})
        self.assertEqual(report["models"]["A"], report["models"]["B"])
        self.assertEqual(report["legs"][1]["sides"], {"left": "B", "right": "A"})
        self.assertEqual(report["summary"]["status"], "complete")

    def test_nonfinite_models_stop_first_leg_in_hard_and_soft(self):
        class NonfiniteModel(torch.nn.Module):
            def forward(self, state, steps=1):
                return torch.full_like(state, float("nan"))
        def bad(*args, **kwargs):
            result = synthetic_bundle(*args, **kwargs)
            result["model"] = NonfiniteModel()
            return result
        self.load.side_effect = bad
        for mode in ("hard", "soft"):
            report = compare.run_comparison(self.args("--mode", mode))
            self.assertEqual(len(report["legs"]), 1)
            self.assertEqual(report["legs"][0]["result"]["tick"], 1)
            self.assertEqual(report["summary"]["status"], "invalid")
            self.assertIsNone(report["summary"]["paired_outcome"])
            json.dumps(report, allow_nan=False)

    def test_failed_model_retains_partial_leg_and_suppresses_winner(self):
        class FailingModel(GrowingModel):
            calls = 0
            def forward(self, state, steps=1):
                self.calls += 1
                if self.calls == 5:
                    raise RuntimeError("bad model at /private/path")
                return super().forward(state, steps)
        def bad(*args, **kwargs):
            result = synthetic_bundle(*args, **kwargs)
            result["model"] = FailingModel()
            return result
        self.load.side_effect = bad
        report = compare.run_comparison(self.args())
        leg = report["legs"][0]
        self.assertEqual(leg["status"], "failed")
        self.assertEqual(leg["result"]["tick"], 4)
        self.assertEqual(leg["result"]["scored_ticks"], 2)
        self.assertEqual(leg["error"], {"stage": "simulation", "type": "RuntimeError"})
        self.assertEqual(len(report["legs"]), 1)
        self.assertIsNone(report["summary"]["paired_outcome"])
        self.assertNotIn("/private/path", json.dumps(report, allow_nan=False))

    def test_load_failure_writes_invalid_report_and_exits_nonzero(self):
        self.load.side_effect = ValueError("No usable checkpoint at /private/path")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "failed.json"
            with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
                code = compare.main(["--report", str(path)])
            self.assertEqual(code, 1)
            report = json.loads(path.read_text())
            self.assertEqual(len(report["legs"]), 1)
            self.assertEqual(report["legs"][0]["error"]["type"], "ValueError")
            self.assertIsNone(report["summary"]["paired_outcome"])
            self.assertIn("no comparison winner", stdout.getvalue())
            self.assertNotIn("/private/path", path.read_text())

    def test_later_failure_preserves_completed_pairs_but_no_overall_winner(self):
        calls = 0
        def sometimes_bad(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise RuntimeError("later model load failed")
            return synthetic_bundle(*args, **kwargs)
        self.load.side_effect = sometimes_bad
        report = compare.run_comparison(self.args())
        self.assertEqual(len(report["legs"]), 3)
        self.assertEqual(report["summary"]["complete_pairs"], 1)
        self.assertIsNone(report["summary"]["scores"])
        self.assertIsNone(report["summary"]["paired_outcome"])
        self.assertEqual(report["summary"]["completed_pair_scores"], report["summary"]["per_seed"][0]["scores"])

    def test_model_identity_drift_stops_before_second_round_steps(self):
        calls = 0
        def changing(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = synthetic_bundle(*args, **kwargs)
            if calls > 2:
                result["seed_dir"] = "seed_999"
            return result
        self.load.side_effect = changing
        report = compare.run_comparison(self.args())
        self.assertEqual(len(report["legs"]), 2)
        self.assertEqual(report["legs"][0]["status"], "complete")
        self.assertEqual(report["legs"][1]["status"], "failed")
        self.assertEqual(report["legs"][1]["result"]["tick"], 0)
        self.assertIsNone(report["summary"]["paired_outcome"])

    def test_cli_report_is_optional_and_stdout_is_concise(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.json"
            for extra in ([], ["--report", str(path)]):
                with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
                    code = compare.main(["--round-ticks", "4", "--warmup-ticks", "1", "--seeds", "7", *extra])
                self.assertEqual(code, 0)
                self.assertIn("Paired score", stdout.getvalue())
                self.assertIn("A: 01_heart [seed_000]; B: 02_star [seed_000]", stdout.getvalue())
                self.assertNotIn('"schema_version"', stdout.getvalue())
                self.assertEqual(path.exists(), bool(extra))
            report = json.loads(path.read_text())
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["summary"]["complete_pairs"], 1)

    def test_display_uses_separators_without_changing_json_scores(self):
        report = compare.run_comparison(self.args("--seeds", "0"))
        report["summary"]["scores"] = {"A": 1234567, "B": 1234568}
        report["summary"]["paired_outcome"] = "B"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            compare.print_summary(report)
        self.assertIn("A=1,234,567, B=1,234,568 cell-ticks", output.getvalue())
        self.assertEqual(report["summary"]["scores"], {"A": 1234567, "B": 1234568})

    def test_no_pygame_or_sdl_import_in_headless_subprocess(self):
        code = f'''
import sys
sys.path.insert(0, {str(V2_ROOT)!r})
sys.path.insert(0, {str(V2_ROOT / 'tests')!r})
class NoPygame:
    def find_spec(self, fullname, *args):
        if fullname == "pygame" or fullname.startswith("pygame."):
            raise AssertionError("headless comparison tried importing pygame")
sys.meta_path.insert(0, NoPygame())
import arena, compare
from test_compare import synthetic_bundle
arena.ensure_model = synthetic_bundle
assert compare.main(["--round-ticks", "3", "--warmup-ticks", "1", "--seeds", "7"]) == 0
assert "pygame" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(os.getenv("PETRI_TEST_CHECKPOINTS") == "1", "opt-in shipped checkpoint comparison")
    def test_short_shipped_checkpoint_pair(self):
        self.loader.stop()
        report = compare.run_comparison(self.args("--seeds", "7", "--round-ticks", "8"))
        self.assertEqual(report["summary"]["status"], "complete")
        self.assertEqual(len(report["legs"]), 2)
        self.assertEqual(report["models"]["A"]["name"], "01_heart")
        self.assertEqual(report["models"]["B"]["name"], "02_star")
        self.assertEqual(report["runtime"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
