"""Runtime and benchmark tests, with no graphics or checkpoint prerequisites."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from benchmark import build_parser, run_benchmark
from runtime import benchmark_steps, configure_runtime, resolve_cpu_threads, seed_all


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()

    def tearDown(self):
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(self.deterministic)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)

    def test_cpu_thread_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_cpu_threads(), 1)
        with patch.dict(os.environ, {"PETRI_CPU_THREADS": "3"}):
            self.assertEqual(resolve_cpu_threads(), 3)
            self.assertEqual(resolve_cpu_threads(2), 2)

    def test_invalid_thread_values(self):
        for value in (0, -1, True, "many", 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_cpu_threads(value)
        with patch.dict(os.environ, {"PETRI_CPU_THREADS": "0"}), self.assertRaises(ValueError):
            resolve_cpu_threads()

    def test_seed_replays_all_three_rngs(self):
        seed_all(42)
        first = (random.random(), np.random.rand(5), torch.rand(5))
        seed_all(42)
        second = (random.random(), np.random.rand(5), torch.rand(5))
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_invalid_seed_does_not_change_threads(self):
        for seed in (-1, 2**32, 3.2, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                configure_runtime(seed=seed, cpu_threads=1)
            self.assertEqual(torch.get_num_threads(), self.threads)

    def test_repeat_configuration_and_metadata(self):
        interop = torch.get_num_interop_threads()
        for _ in range(2):
            metadata = configure_runtime(seed=7, cpu_threads=1)
            self.assertEqual(metadata["cpu_threads"], 1)
            self.assertEqual(metadata["device"], "cpu")
            self.assertEqual(metadata["seed"], 7)
            self.assertTrue(metadata["deterministic_algorithms"])
            self.assertEqual(metadata["interop_threads"], interop)
            json.dumps(metadata, allow_nan=False)

    def test_warmup_is_excluded_from_samples(self):
        calls = []

        def step():
            self.assertTrue(torch.is_inference_mode_enabled())
            calls.append(1)

        # The measured calls take 1, 2, 3, 4 milliseconds. Warmup isn't timed.
        times = [0, 1_000_000, 2_000_000, 4_000_000, 5_000_000, 8_000_000, 9_000_000, 13_000_000]
        with patch("runtime.time.perf_counter_ns", side_effect=times):
            result = benchmark_steps(step, warmup=3, steps=4)
        self.assertEqual(len(calls), 7)
        self.assertEqual(result["step_ms"]["p50"], 2.5)
        self.assertAlmostEqual(result["step_ms"]["p95"], 3.85)
        self.assertEqual(result["total_measured_ms"], 10)
        self.assertEqual(result["steps_per_second"], 400)

    def test_invalid_benchmark_counts(self):
        for kwargs in ({"warmup": -1}, {"warmup": True}, {"steps": 0}, {"steps": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                benchmark_steps(lambda: None, **kwargs)

    def test_synthetic_replay_all_modes(self):
        for mode in ("nca", "soft", "hard"):
            with self.subTest(mode=mode):
                args = build_parser().parse_args([
                    "--synthetic", "--mode", mode, "--grid-size", "8", "--channels", "8",
                    "--hidden-size", "8", "--cpu-threads", "1", "--warmup", "1", "--steps", "2",
                ])
                first = run_benchmark(args)
                second = run_benchmark(args)
                self.assertEqual(first["final_state"], second["final_state"])
                self.assertEqual(first["runtime"]["device"], "cpu")
                self.assertEqual(first["measured_steps"], 2)
                self.assertTrue(all(state["finite"] for state in first["final_state"]))

    def test_cli_stdout_is_json(self):
        result = subprocess.run(
            [sys.executable, str(V2_ROOT / "benchmark.py"), "--synthetic", "--mode", "nca",
             "--grid-size", "8", "--hidden-size", "8", "--warmup", "0", "--steps", "1"],
            check=True, capture_output=True, text=True,
        )
        report = json.loads(result.stdout)
        self.assertTrue(report["synthetic"])
        self.assertEqual(report["models"][0]["kind"], "synthetic-untrained")
        self.assertIn("p95", report["step_ms"])


if __name__ == "__main__":
    unittest.main()
