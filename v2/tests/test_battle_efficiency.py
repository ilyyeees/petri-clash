"""Fused cell masks preserve the original battle state, observers and errors."""

from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import battle


def original_step(a, b, owner, control, model_a, model_b, *, strict_finite=False):
    """Independent pre-fusion expressions, with the default public rules."""
    def finite_cells(state):
        finite = torch.isfinite(state).all(dim=1, keepdim=True)
        if strict_finite and not bool(finite.all()):
            raise FloatingPointError("Non-finite NCA state or proposal")
        return torch.where(finite, state, 0.0)

    def near(mask):
        return F.max_pool2d(mask.to(torch.float32), 3, stride=1, padding=1) > 0

    with torch.inference_mode():
        if strict_finite and not bool(torch.isfinite(control).all()):
            raise FloatingPointError("Non-finite battle control")
        owned_a, owned_b = owner == 1, owner == 2
        clean_a = torch.where(owned_b, 0.0, finite_cells(a))
        clean_b = torch.where(owned_a, 0.0, finite_cells(b))
        support_a = near((clean_a[:, 3:4] > .1) | owned_a)
        support_b = near((clean_b[:, 3:4] > .1) | owned_b)
        proposed_a = model_a(clean_a, steps=1)
        proposed_b = model_b(clean_b, steps=1)
        proposed_a = torch.where(support_a, finite_cells(proposed_a), 0.0)
        proposed_b = torch.where(support_b, finite_cells(proposed_b), 0.0)
        alpha_a = proposed_a[:, 3:4].clamp(0.0, 1.0)
        alpha_b = proposed_b[:, 3:4].clamp(0.0, 1.0)
        strength_a = torch.where(alpha_a > .05, alpha_a, 0.0)
        strength_b = torch.where(alpha_b > .05, alpha_b, 0.0)
        pressure = strength_a - strength_b
        pressure = torch.where(pressure.abs() >= .01, pressure, 0.0)
        clean_control = torch.nan_to_num(control, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        next_control = (clean_control * .97 + pressure * .28).clamp(-1.0, 1.0)
        next_owner = torch.where(
            ((owner == 1) & (next_control <= .12)) | ((owner == 2) & (next_control >= -.12)),
            torch.zeros_like(owner), owner)
        next_owner = torch.where(next_control >= .55, torch.ones_like(owner), next_owner)
        next_owner = torch.where(next_control <= -.55, torch.full_like(owner, 2), next_owner)
        next_a = torch.where(next_owner != 2, proposed_a, 0.0)
        next_b = torch.where(next_owner != 1, proposed_b, 0.0)
        life_a = near(next_a[:, 3:4] > .1)
        life_b = near(next_b[:, 3:4] > .1)
        next_a = torch.where(life_a, next_a, 0.0)
        next_b = torch.where(life_b, next_b, 0.0)
        abandoned = ((next_owner == 1) & ~life_a) | ((next_owner == 2) & ~life_b)
        next_owner = torch.where(abandoned, torch.zeros_like(next_owner), next_owner)
        next_control = torch.where(abandoned | ~(life_a | life_b), 0.0, next_control)
    return next_a, next_b, next_owner, next_control


def with_layout(value, layout):
    """Change physical layout while preserving shape and all logical values."""
    if layout == "channels_last":
        return value.contiguous(memory_format=torch.channels_last)
    if layout == "transposed":
        return value.transpose(-1, -2).contiguous().transpose(-1, -2)
    if layout == "batch_spatial":
        return value.permute(2, 0, 3, 1).contiguous().permute(1, 3, 0, 2)
    if layout == "contiguous":
        return value.contiguous()
    return value


class RetainedProposal:
    def __init__(self, unstable=False, alias=False, output_layout="unchanged"):
        self.unstable, self.alias = unstable, alias
        self.output_layout = output_layout

    def __call__(self, state, steps=1):
        self.input = state
        self.input_copy = state.clone()
        self.output = state if self.alias else state + torch.rand_like(state) * .02
        if self.unstable:
            self.output[:, 4, 0, 0] = float("nan")
            self.output[:, 4, -1, -1] = float("inf")
        self.output = with_layout(self.output, self.output_layout)
        self.output_copy = self.output.clone()
        return self.output


class BattleEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.rng = torch.get_rng_state()
        torch.set_num_threads(1)
        torch.manual_seed(1729)

    def tearDown(self):
        torch.set_num_threads(self.threads)
        torch.set_rng_state(self.rng)

    def assert_bits(self, actual, expected):
        self.assertEqual((actual.shape, actual.dtype, actual.stride()),
                         (expected.shape, expected.dtype, expected.stride()))
        self.assertTrue(torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                                    expected.contiguous().reshape(-1).view(torch.uint8)))

    def world(self, dtype=torch.float32, owner_dtype=torch.int64, layout="contiguous"):
        a, b = (torch.randn(2, channels, 9, 13, dtype=dtype) for channels in (6, 8))
        # Preserve both zero signs in retained and removed hidden channels.
        a[:, 5, ::2] = -0.0
        b[:, 5, ::2] = 0.0
        owner = torch.randint(0, 3, (2, 1, 9, 13), dtype=owner_dtype)
        control = torch.randn(2, 1, 9, 13, dtype=dtype)
        values = a, b, owner, control
        if layout == "channels_last":
            return tuple(v.contiguous(memory_format=torch.channels_last) for v in values)
        if layout == "transposed":
            return tuple(v.transpose(-1, -2) for v in values)
        return values

    def test_exact_state_rng_layout_and_input_observers(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for owner_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                for layout in ("contiguous", "channels_last", "transposed"):
                    for alias in (False, True):
                        with self.subTest(dtype=dtype, owner=owner_dtype, layout=layout, alias=alias):
                            world = self.world(dtype, owner_dtype, layout)
                            saved = tuple(v.clone() for v in world)
                            rng = torch.get_rng_state()
                            baseline_models = RetainedProposal(alias=alias), RetainedProposal(alias=alias)
                            expected = original_step(*world, *baseline_models)
                            expected_rng = torch.get_rng_state()
                            torch.set_rng_state(rng)
                            models = RetainedProposal(alias=alias), RetainedProposal(alias=alias)
                            actual = battle.clash_step(*world, *models)
                            for a, b in zip(actual, expected):
                                self.assert_bits(a, b)
                            for a, b in zip(world, saved):
                                self.assert_bits(a, b)
                            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                            for model, baseline_model in zip(models, baseline_models):
                                self.assert_bits(model.input, baseline_model.input)
                                self.assert_bits(model.output, baseline_model.output)
                                self.assert_bits(model.input, model.input_copy)
                                self.assert_bits(model.output, model.output_copy)

    def test_independent_input_and_proposal_layouts_preserve_model_observations(self):
        # A combined mask must retain the original outer where's iteration
        # layout. Models can observe strides and rand_like visits values in
        # that layout, so matching only final logical masks is insufficient.
        source = self.world()
        layouts = ("contiguous", "channels_last", "transposed")
        for input_layouts in product(layouts, repeat=4):
            world = tuple(with_layout(value, layout)
                          for value, layout in zip(source, input_layouts))
            saved = tuple(value.clone() for value in world)
            for output_layout in ("unchanged", "channels_last", "transposed"):
                with self.subTest(inputs=input_layouts, proposal=output_layout):
                    rng = torch.get_rng_state()
                    baseline_models = tuple(RetainedProposal(output_layout=output_layout)
                                            for _ in range(2))
                    expected = original_step(*world, *baseline_models)
                    expected_rng = torch.get_rng_state()
                    torch.set_rng_state(rng)
                    models = tuple(RetainedProposal(output_layout=output_layout)
                                   for _ in range(2))
                    actual = battle.clash_step(*world, *models)
                    for value, baseline_value in zip(actual, expected):
                        self.assert_bits(value, baseline_value)
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                    for model, baseline_model in zip(models, baseline_models):
                        self.assert_bits(model.input, baseline_model.input)
                        self.assert_bits(model.output, baseline_model.output)
                        self.assert_bits(model.input, model.input_copy)
                        self.assert_bits(model.output, model.output_copy)
                    for value, saved_value in zip(world, saved):
                        self.assert_bits(value, saved_value)

    def assert_exact_world(self, world, output_layouts=("unchanged", "unchanged")):
        # Copies may densify an expanded view, so compare its values and its
        # original strides separately when checking caller-owned inputs.
        saved = tuple((value.clone(), value.stride()) for value in world)
        rng = torch.get_rng_state()
        baseline_models = tuple(RetainedProposal(output_layout=layout) for layout in output_layouts)
        expected = original_step(*world, *baseline_models)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        models = tuple(RetainedProposal(output_layout=layout) for layout in output_layouts)
        actual = battle.clash_step(*world, *models)
        for value, baseline_value in zip(actual, expected):
            self.assert_bits(value, baseline_value)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for model, baseline_model in zip(models, baseline_models):
            self.assert_bits(model.input, baseline_model.input)
            self.assert_bits(model.output, baseline_model.output)
            self.assert_bits(model.input, model.input_copy)
            self.assert_bits(model.output, model.output_copy)
        for value, (copy, stride) in zip(world, saved):
            self.assertEqual(value.stride(), stride)
            self.assertTrue(torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
                                        copy.contiguous().reshape(-1).view(torch.uint8)))

    def test_singleton_spatial_layouts_preserve_exact_output_strides(self):
        for batch, height, width in product((1, 2), (1, 3), (1, 3)):
            if height > 1 and width > 1:
                continue
            for channels in (4, 5):
                with self.subTest(batch=batch, height=height, width=width, channels=channels):
                    world = (
                        torch.randn(batch, channels, height, width),
                        with_layout(torch.randn(batch, channels + 1, height, width,
                                                dtype=torch.float64), "channels_last"),
                        with_layout(torch.randint(0, 3, (batch, 1, height, width),
                                                  dtype=torch.int16), "transposed"),
                        with_layout(torch.randn(batch, 1, height, width,
                                                dtype=torch.float64), "batch_spatial"),
                    )
                    self.assert_exact_world(world, ("batch_spatial", "transposed"))

    def test_expanded_state_and_permuted_masks_preserve_rng_value_order(self):
        for height, width in ((1, 1), (1, 3), (3, 1), (3, 5)):
            for expanded_side in (0, 1):
                with self.subTest(height=height, width=width, expanded_side=expanded_side):
                    states = [torch.full((2, channels, height, width), .5)
                              for channels in (5, 6)]
                    states[expanded_side] = states[expanded_side][:1].expand_as(states[expanded_side])
                    owner = with_layout(torch.zeros(2, 1, height, width, dtype=torch.int64),
                                        "batch_spatial")
                    control = torch.zeros(2, 1, height, width)
                    self.assert_exact_world((*states, owner, control))

    def test_nonfinite_quarantine_remains_exact_before_and_after_proposals(self):
        for layout in ("contiguous", "channels_last", "transposed"):
            with self.subTest(layout=layout):
                world = self.world(layout=layout)
                for value in (world[0], world[1], world[3]):
                    value[..., 0, 0] = float("nan")
                    value[..., 0, 1] = float("inf")
                    value[..., 0, 2] = -float("inf")
                rng = torch.get_rng_state()
                expected = original_step(*world, RetainedProposal(unstable=True), RetainedProposal(unstable=True))
                expected_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                models = RetainedProposal(unstable=True), RetainedProposal(unstable=True)
                actual = battle.clash_step(*world, *models)
                for a, b in zip(actual, expected):
                    self.assert_bits(a, b)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                for model in models:
                    self.assert_bits(model.output, model.output_copy)

    def test_strict_validation_does_not_hide_excluded_nonfinite_cells(self):
        for side in (0, 1):
            for proposal in (False, True):
                for nonfinite in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(side=side, proposal=proposal, value=nonfinite):
                        world = self.world()
                        for value in world:
                            value.zero_()
                        world[2][..., 0, 0] = 2 if side == 0 else 1
                        models = [RetainedProposal(alias=True), RetainedProposal(alias=True)]
                        if proposal:
                            class Unstable:
                                def __call__(self, state, steps=1):
                                    result = state.clone()
                                    result[:, 4, -1, -1] = nonfinite
                                    return result
                            models[side] = Unstable()
                        else:
                            world[side][:, 4, 0, 0] = nonfinite
                        for function in (original_step, battle.clash_step):
                            with self.assertRaisesRegex(FloatingPointError, "Non-finite NCA state or proposal"):
                                function(*world, *models, strict_finite=True)

    def test_canonical_cpu_grid_requires_exact_positive_dense_strides(self):
        self.assertTrue(battle._canonical_cpu_grid(torch.empty(2, 4, 3, 5)))
        self.assertTrue(battle._canonical_cpu_grid(torch.empty(1, 1, 1, 1)))
        values = (
            torch.empty_strided((1, 4, 3, 5), (100, 15, 5, 1)),
            torch.empty_strided((2, 1, 3, 5), (15, 2, 5, 1)),
            torch.empty_strided((2, 4, 1, 5), (20, 5, 7, 1)),
            torch.empty_strided((2, 4, 3, 1), (12, 3, 1, 7)),
            torch.empty(2, 4, 3, 5).contiguous(memory_format=torch.channels_last),
            torch.empty(1, 4, 3, 5).expand(2, -1, -1, -1),
            torch.empty(0, 4, 3, 5),
            torch.empty(2, 4, 3, 5, device="meta"),
            torch.empty(4),
        )
        for value in values:
            with self.subTest(shape=value.shape, stride=value.stride(), device=value.device):
                self.assertFalse(battle._canonical_cpu_grid(value))

    def test_canonical_cpu_fuses_where_passes_and_compiler_falls_back(self):
        world = self.world()
        actual_where = torch.where
        shapes = []
        def observe(*args, **kwargs):
            result = actual_where(*args, **kwargs)
            if isinstance(result, torch.Tensor) and result.ndim == 4 and result.shape[1] > 1:
                shapes.append(tuple(result.shape))
            return result
        rng = torch.get_rng_state()
        expected = original_step(*world, RetainedProposal(), RetainedProposal())
        expected_rng = torch.get_rng_state()
        for function, compiling, count in (
                (original_step, False, 12), (battle.clash_step, False, 6),
                (battle.clash_step, True, 12)):
            with self.subTest(function=function.__name__, compiling=compiling):
                shapes.clear()
                torch.set_rng_state(rng)
                with patch("battle.torch.compiler.is_compiling", return_value=compiling), \
                        patch("battle.torch.where", side_effect=observe):
                    actual = function(*world, RetainedProposal(), RetainedProposal())
                self.assertEqual(len(shapes), count)
                for value, baseline_value in zip(actual, expected):
                    self.assert_bits(value, baseline_value)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))


if __name__ == "__main__":
    unittest.main()
