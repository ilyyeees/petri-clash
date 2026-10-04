"""Finite-cell reduction preserves quarantine, bits, layout and strict errors."""

from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import battle


def original_finite_cells(state, strict=False, keep=None):
    """Independent pre-optimization finite predicate and original where order."""
    finite = torch.isfinite(state).all(dim=1, keepdim=True)
    if strict and not bool(finite.all()):
        raise FloatingPointError("Non-finite NCA state or proposal")
    result = torch.where(finite, state, 0.0)
    return result if keep is None else torch.where(keep, result, 0.0)


def with_layout(value, layout):
    if layout == "channels_last":
        return value.contiguous(memory_format=torch.channels_last)
    if layout == "transposed":
        return value.transpose(-1, -2).contiguous().transpose(-1, -2)
    return value.contiguous()


class FiniteEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.rng = torch.get_rng_state()
        torch.set_num_threads(1)
        torch.manual_seed(913)

    def tearDown(self):
        torch.set_num_threads(self.threads)
        torch.set_rng_state(self.rng)

    def assert_bits(self, actual, expected):
        self.assertEqual((actual.shape, actual.dtype, actual.stride()),
                         (expected.shape, expected.dtype, expected.stride()))
        # Flatten first: an otherwise contiguous tensor can retain a non-unit
        # final stride when the final dimension has length one.
        self.assertTrue(torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8)))

    def check_case(self, state, keep=None, strict=False):
        before = state.clone(memory_format=torch.preserve_format)
        keep_before = None if keep is None else keep.clone(memory_format=torch.preserve_format)
        rng = torch.get_rng_state()
        expected = original_finite_cells(state, strict, keep)
        actual = battle._finite_cells(state, strict, keep)
        self.assert_bits(actual, expected)
        self.assert_bits(state, before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        if keep is not None:
            self.assert_bits(keep, keep_before)

    def test_nonfinite_extremes_and_actual_subnormals_in_every_channel(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            info = torch.finfo(dtype)
            smallest = torch.nextafter(torch.tensor(0.0, dtype=dtype),
                                       torch.tensor(1.0, dtype=dtype)).item()
            # Include both tiny/2 and the smallest subnormal. Neither is zero.
            values = (float("nan"), float("inf"), -float("inf"), 0.0, -0.0,
                      1.0, -1.0, info.max, -info.max, info.tiny, -info.tiny,
                      info.tiny / 2, -info.tiny / 2, smallest, -smallest)
            triples = torch.tensor(list(product(values, repeat=3)), dtype=dtype)
            self.assertGreater(smallest, 0.0)
            self.assertGreater(torch.tensor(info.tiny / 2, dtype=dtype).item(), 0.0)
            for channel in range(8):
                state = torch.zeros(1, 8, 15, 225, dtype=dtype)
                for offset in range(3):
                    state[0, (channel + offset) % 8] = triples[:, offset].reshape(15, 225)
                for layout in ("contiguous", "channels_last", "transposed"):
                    with self.subTest(dtype=dtype, channel=channel, layout=layout):
                        value = with_layout(state, layout)
                        self.check_case(value)
                        keep = torch.ones_like(value[:, :1], dtype=torch.bool)
                        keep[..., ::2] = False
                        self.check_case(value, keep)

    def test_all_half_and_bfloat16_bit_patterns_in_every_channel(self):
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        for dtype in (torch.float16, torch.bfloat16):
            values = bits.view(dtype).reshape(256, 256)
            for channel in range(8):
                with self.subTest(dtype=dtype, channel=channel):
                    state = torch.ones(1, 8, 256, 256, dtype=dtype)
                    state[0, channel] = values
                    self.check_case(state)

    def test_random_full_float32_and_float64_bit_patterns(self):
        for dtype, int_dtype, low, high in (
            (torch.float32, torch.int32, -(2**31), 2**31),
            (torch.float64, torch.int64, -(2**63), 2**63 - 1),
        ):
            with self.subTest(dtype=dtype):
                state = torch.randint(low, high, (2, 24, 64, 64), dtype=int_dtype).view(dtype)
                self.check_case(state)

    def test_keep_and_state_layouts_vary_independently(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            state = torch.randn(2, 8, 9, 13, dtype=dtype)
            state[:, 4, ::2] = -0.0
            state[:, 5, 0, 0] = float("nan")
            keep = torch.rand(2, 1, 9, 13) > 0.5
            for state_layout, keep_layout in product(
                    ("contiguous", "channels_last", "transposed"), repeat=2):
                with self.subTest(dtype=dtype, state=state_layout, keep=keep_layout):
                    self.check_case(with_layout(state, state_layout),
                                    with_layout(keep, keep_layout))

    def test_strict_checks_even_nonfinite_cells_excluded_by_keep(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for channel in range(8):
                for nonfinite in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(dtype=dtype, channel=channel, value=nonfinite):
                        state = torch.zeros(2, 8, 7, 11, dtype=dtype)
                        state[1, channel, 3, 5] = nonfinite
                        before = state.clone()
                        keep = torch.zeros(2, 1, 7, 11, dtype=torch.bool)
                        for function in (original_finite_cells, battle._finite_cells):
                            with self.assertRaisesRegex(
                                    FloatingPointError, "Non-finite NCA state or proposal"):
                                function(state, strict=True, keep=keep)
                        self.assert_bits(state, before)

    def test_strict_finite_extremes_preserve_every_bit(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                state = torch.zeros(2, 8, 7, 11, dtype=dtype)
                state[:, 0] = torch.finfo(dtype).max
                state[:, 1] = -torch.finfo(dtype).max
                state[:, 2] = torch.finfo(dtype).tiny / 2
                state[:, 4] = -0.0
                self.check_case(state, strict=True)

    def test_compiler_and_unsupported_cpu_inputs_use_original_predicate(self):
        real_isfinite = torch.isfinite
        cases = (
            (torch.randn(2, 8, 7, 11), True),
            (torch.ones(2, 8, 7, 11, dtype=torch.int64), False),
            (torch.zeros(1, 0, 7, 11), False),
        )
        for state, compiling in cases:
            with self.subTest(dtype=state.dtype, shape=state.shape, compiling=compiling):
                expected = original_finite_cells(state)
                with patch("battle.torch.compiler.is_compiling", return_value=compiling), \
                        patch("battle.torch.isfinite", wraps=real_isfinite) as isfinite:
                    actual = battle._finite_cells(state)
                self.assert_bits(actual, expected)
                self.assertEqual(isfinite.call_count, 1)
                self.assertIs(isfinite.call_args.args[0], state)

    def test_non_cpu_meta_uses_original_predicate_without_gpu_execution(self):
        # Meta performs shape dispatch only. This asserts the non-CPU fallback;
        # it does not claim to execute or validate a GPU kernel.
        state = torch.empty(2, 8, 7, 11, device="meta")
        expected = original_finite_cells(state)
        with patch("battle.torch.isfinite", wraps=torch.isfinite) as isfinite:
            actual = battle._finite_cells(state)
        self.assertEqual(isfinite.call_count, 1)
        self.assertIs(isfinite.call_args.args[0], state)
        self.assertEqual((actual.shape, actual.dtype, actual.device, actual.stride()),
                         (expected.shape, expected.dtype, expected.device, expected.stride()))

    def test_empty_and_singleton_dimensions_retain_original_behavior(self):
        for shape in ((0, 8, 7, 11), (1, 8, 0, 11), (1, 8, 7, 0),
                      (1, 8, 1, 1), (1, 0, 7, 11)):
            with self.subTest(shape=shape):
                self.check_case(torch.zeros(shape), strict=True)


if __name__ == "__main__":
    unittest.main()
