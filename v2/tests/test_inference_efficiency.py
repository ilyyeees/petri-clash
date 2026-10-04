"""The CPU inference activation reuse must preserve NCA math and observers."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nca import NCA


_ORIGINAL_RELU = F.relu


def original_step(model, state, fire_rate):
    """Independent copy of the pre-optimization step, without NCA helpers."""
    pre_life = F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > 0.1
    perceived = F.conv2d(
        state, model.perception_filters, padding=1, groups=model.channels
    )
    delta = model.fc1(_ORIGINAL_RELU(model.fc0(perceived)))
    if fire_rate < 1.0:
        mask = (torch.rand_like(state[:, :1]) <= fire_rate).float()
        delta = delta * mask
    state = state + delta
    post_life = F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > 0.1
    return state * (pre_life | post_life).float()


def original_forward(model, state, steps, fire_rate):
    for _ in range(steps):
        state = original_step(model, state, fire_rate)
    return state


class CachedConv2d(nn.Conv2d):
    """A legal Conv2d subclass whose result is not a fresh private tensor."""

    def forward(self, state):
        if not hasattr(self, "cached_output"):
            self.cached_output = super().forward(state)
        return self.cached_output


class CachedProjection(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.cached_output = output

    def forward(self, state):
        return self.cached_output


class InferenceEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.rng = torch.get_rng_state()
        torch.set_num_threads(1)
        torch.manual_seed(1729)

    def tearDown(self):
        torch.set_num_threads(self.threads)
        torch.set_rng_state(self.rng)

    def make_model(self):
        model = NCA(channels=8, hidden_size=12)
        with torch.no_grad():
            # Loaded checkpoints need not contain repeated perception filters,
            # and zero-initialized fc1 would hide almost any activation bug.
            model.perception_filters.copy_(
                torch.randn_like(model.perception_filters) * 0.1
            )
            model.fc0.bias.copy_(torch.linspace(-0.4, 0.4, model.hidden_size))
            model.fc1.weight.normal_(std=0.025)
            model.fc1.bias.normal_(std=0.01)
        return model

    def make_state(self, layout="contiguous"):
        state = torch.randn(2, 8, 7, 11) * 0.1
        state[:, 3] = 0
        state[:, 3, 2:5, 3:6] = 0.9
        if layout == "channels_last":
            return state.contiguous(memory_format=torch.channels_last)
        if layout == "transposed":
            return state.transpose(-1, -2).contiguous().transpose(-1, -2)
        return state

    def assert_exact(self, actual, expected, *, layout=True):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        if layout:
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(
                actual.is_contiguous(memory_format=torch.channels_last),
                expected.is_contiguous(memory_format=torch.channels_last),
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(
            actual.detach().contiguous().view(torch.uint8),
            expected.detach().contiguous().view(torch.uint8),
        ), "Tensor bytes changed")

    def assert_relu_mode(self, relu, inplace, calls=1):
        self.assertEqual(relu.call_count, calls)
        self.assertEqual(
            [call.kwargs.get("inplace", False) for call in relu.call_args_list],
            [inplace] * calls,
        )

    def check_reference_matrix(self, autocast=False):
        for layout in ("contiguous", "channels_last", "transposed"):
            for fire_rate in (0.5, 1.0):
                for steps in (1, 4):
                    with self.subTest(layout=layout, fire_rate=fire_rate, steps=steps):
                        model = self.make_model().eval()
                        state = self.make_state(layout)
                        saved_state = state.clone(memory_format=torch.preserve_format)
                        state_pointer = state.data_ptr()
                        saved_model = {
                            name: value.clone() for name, value in model.state_dict().items()
                        }
                        model_pointers = {
                            name: value.data_ptr() for name, value in model.state_dict().items()
                        }
                        initial_rng = torch.get_rng_state()
                        precision = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
                        with torch.inference_mode(), precision:
                            expected = original_forward(model, state, steps, fire_rate)
                            expected_rng = torch.get_rng_state()
                            torch.set_rng_state(initial_rng)
                            with patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                                actual = model(state, steps=steps, fire_rate=fire_rate)
                            self.assert_relu_mode(relu, True, calls=steps)
                        self.assert_exact(actual, expected)
                        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                        self.assert_exact(state, saved_state)
                        self.assertEqual(state.data_ptr(), state_pointer)
                        self.assertEqual(set(model.state_dict()), set(saved_model))
                        for name, value in model.state_dict().items():
                            self.assert_exact(value, saved_model[name])
                            self.assertEqual(value.data_ptr(), model_pointers[name])

    def test_cpu_inference_matches_original_values_bytes_rng_layout_and_storage(self):
        self.check_reference_matrix()

    def test_cpu_bfloat16_autocast_matches_original(self):
        # Probe autocast itself separately so a failure in NCA is never skipped.
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                probe = F.conv2d(torch.ones(1, 1, 3, 3), torch.ones(1, 1, 1, 1))
        except (RuntimeError, TypeError) as error:
            self.skipTest(f"CPU bfloat16 autocast unavailable: {error}")
        if probe.dtype != torch.bfloat16:
            self.skipTest("CPU convolution does not use bfloat16 autocast")
        self.check_reference_matrix(autocast=True)

    def test_only_eval_cpu_inference_mode_uses_inplace_relu(self):
        for training in (True, False):
            for mode in ("grad", "no_grad", "inference"):
                with self.subTest(training=training, mode=mode):
                    model = self.make_model().train(training)
                    state = self.make_state().requires_grad_()
                    context = {
                        "grad": torch.enable_grad,
                        "no_grad": torch.no_grad,
                        "inference": torch.inference_mode,
                    }[mode]
                    with context(), patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                        actual = model(state, fire_rate=1.0)
                    self.assert_relu_mode(relu, not training and mode == "inference")
                    self.assertEqual(actual.requires_grad, mode == "grad")

    def test_gradients_match_original_in_train_and_eval(self):
        for training in (True, False):
            for fire_rate in (0.5, 1.0):
                with self.subTest(training=training, fire_rate=fire_rate):
                    model = self.make_model().train(training)
                    reference = deepcopy(model)
                    state = self.make_state().requires_grad_()
                    reference_state = state.detach().clone().requires_grad_()
                    initial_rng = torch.get_rng_state()
                    expected = original_forward(reference, reference_state, 3, fire_rate)
                    expected.square().sum().backward()
                    expected_rng = torch.get_rng_state()
                    torch.set_rng_state(initial_rng)
                    actual = model(state, steps=3, fire_rate=fire_rate)
                    actual.square().sum().backward()
                    self.assert_exact(actual, expected)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assert_exact(state.grad, reference_state.grad)
                    for (name, parameter), (reference_name, original) in zip(
                        model.named_parameters(), reference.named_parameters()
                    ):
                        self.assertEqual(name, reference_name)
                        self.assertIsNotNone(parameter.grad)
                        self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
                        self.assert_exact(parameter.grad, original.grad)

    def register_hook(self, model, hook, scope):
        if scope == "global":
            def only_fc0(module, args, output):
                if module is model.fc0:
                    return hook(module, args, output)
                return None
            return nn.modules.module.register_module_forward_hook(only_fc0)
        return model.fc0.register_forward_hook(hook)

    def test_local_and_global_hooks_preserve_retained_preactivations(self):
        for scope in ("local", "global"):
            with self.subTest(scope=scope):
                model = self.make_model().eval()
                state = self.make_state()
                retained = []

                def retain(module, args, output):
                    retained.append((output, output.clone()))

                handle = self.register_hook(model, retain, scope)
                try:
                    with torch.inference_mode(), patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                        model(state, fire_rate=1.0)
                    self.assert_relu_mode(relu, False)
                    self.assertEqual(len(retained), 1)
                    activation, before = retained[0]
                    self.assertTrue((before < 0).any())
                    self.assert_exact(activation, before)
                finally:
                    handle.remove()

    def test_local_and_global_hook_replacements_preserve_external_storage(self):
        for scope in ("local", "global"):
            with self.subTest(scope=scope):
                model = self.make_model().eval()
                state = self.make_state()
                external = torch.linspace(-1, 1, 2 * 12 * 7 * 11).reshape(2, 12, 7, 11)
                before = external.clone()
                pointer = external.data_ptr()

                def replace(module, args, output):
                    return external.view_as(output)

                handle = self.register_hook(model, replace, scope)
                try:
                    with torch.inference_mode():
                        expected = original_forward(model, state, 2, 1.0)
                        with patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                            actual = model(state, steps=2, fire_rate=1.0)
                    self.assert_relu_mode(relu, False, calls=2)
                    self.assert_exact(actual, expected)
                    self.assert_exact(external, before)
                    self.assertEqual(external.data_ptr(), pointer)
                finally:
                    handle.remove()

    def test_self_removing_hooks_are_checked_before_fc0_runs(self):
        for scope in ("local", "global"):
            for replace_output in (False, True):
                with self.subTest(scope=scope, replace_output=replace_output):
                    model = self.make_model().eval()
                    state = self.make_state()
                    external = torch.linspace(-1, 1, 2 * 12 * 7 * 11).reshape(2, 12, 7, 11)
                    external_before = external.clone()
                    retained = []

                    def remove_self(module, args, output):
                        retained.append((output, output.clone()))
                        handle.remove()
                        if replace_output:
                            return external.view_as(output)
                        return None

                    handle = self.register_hook(model, remove_self, scope)
                    try:
                        with torch.inference_mode():
                            with patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                                model(state, fire_rate=1.0)
                            self.assert_relu_mode(relu, False)
                            with patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                                model(state, fire_rate=1.0)
                            self.assert_relu_mode(relu, True)
                        self.assertEqual(len(retained), 1)
                        self.assertTrue((retained[0][1] < 0).any())
                        self.assert_exact(*retained[0])
                        self.assert_exact(external, external_before)
                    finally:
                        handle.remove()

    def test_custom_fc0_variants_keep_cached_outputs(self):
        for layer_kind in ("conv_subclass", "replacement", "instance_forward"):
            with self.subTest(layer_kind=layer_kind):
                model = self.make_model().eval()
                state = self.make_state()
                with torch.inference_mode():
                    if layer_kind == "conv_subclass":
                        layer = CachedConv2d(24, 12, 1)
                        layer.load_state_dict(model.fc0.state_dict())
                        cached = layer(model.perceive(state))
                    else:
                        cached = model.fc0(model.perceive(state))
                        if layer_kind == "instance_forward":
                            layer = model.fc0
                            layer.forward = lambda state, cached=cached: cached
                            self.assertIs(type(layer), nn.Conv2d)
                        else:
                            layer = CachedProjection(cached)
                    model.fc0 = layer.eval()
                    before = cached.clone()
                    self.assertTrue((before < 0).any())
                    expected = original_forward(model, state, 2, 1.0)
                    with patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                        actual = model(state, steps=2, fire_rate=1.0)
                self.assert_relu_mode(relu, False, calls=2)
                self.assert_exact(actual, expected)
                self.assert_exact(cached, before)

    def test_pre_hooks_can_install_self_removing_output_hooks(self):
        for scope in ("local", "global"):
            for replace_output in (False, True):
                with self.subTest(scope=scope, replace_output=replace_output):
                    model = self.make_model().eval()
                    state = self.make_state()
                    external = torch.linspace(-1, 1, 2 * 12 * 7 * 11).reshape(2, 12, 7, 11)
                    external_before = external.clone()
                    retained, output_handles = [], []

                    def install_output_hook(module, args):
                        if module is not model.fc0:
                            return None

                        def remove_self(module, args, output):
                            retained.append((output, output.clone()))
                            output_handle.remove()
                            return external.view_as(output) if replace_output else None

                        output_handle = module.register_forward_hook(remove_self)
                        output_handles.append(output_handle)
                        return None

                    if scope == "global":
                        pre_handle = nn.modules.module.register_module_forward_pre_hook(install_output_hook)
                    else:
                        pre_handle = model.fc0.register_forward_pre_hook(install_output_hook)
                    try:
                        with torch.inference_mode():
                            expected = original_step(model, state, 1.0)
                            retained.clear()
                            with patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                                actual = model(state, fire_rate=1.0)
                            self.assert_relu_mode(relu, False)
                            self.assertFalse(model.fc0._forward_hooks)
                            pre_handle.remove()
                            with patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                                model(state, fire_rate=1.0)
                            self.assert_relu_mode(relu, True)
                        self.assert_exact(actual, expected)
                        self.assertEqual(len(retained), 1)
                        self.assertTrue((retained[0][1] < 0).any())
                        self.assert_exact(*retained[0])
                        self.assert_exact(external, external_before)
                    finally:
                        pre_handle.remove()
                        for handle in output_handles:
                            handle.remove()

    def test_unknown_global_hook_registry_fails_closed(self):
        model = self.make_model().eval()
        state = self.make_state()
        # Hide only the registry seen by the guard; removing PyTorch's real
        # registry would break Module.__call__ before reaching the activation.
        registries = ("_global_forward_hooks", "_global_forward_pre_hooks")
        for missing in registries:
            with self.subTest(missing=missing):
                registry = SimpleNamespace(**{name: {} for name in registries if name != missing})
                unknown_registry = SimpleNamespace(
                    Conv2d=nn.Conv2d, modules=SimpleNamespace(module=registry)
                )
                with torch.inference_mode(), patch("nca.nn", unknown_registry), \
                        patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
                    model(state, fire_rate=1.0)
                self.assert_relu_mode(relu, False)

    def test_compiler_capture_uses_out_of_place_relu(self):
        model = self.make_model().eval()
        state = self.make_state()
        with torch.inference_mode(), patch("torch.compiler.is_compiling", return_value=True), \
                patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
            model(state, fire_rate=1.0)
        self.assert_relu_mode(relu, False)

    def test_eval_fullgraph_eager_compilation_matches_original(self):
        model = self.make_model().eval()
        state = self.make_state()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.addCleanup(torch.compiler.reset)
        with torch.inference_mode():
            initial_rng = torch.get_rng_state()
            expected = original_forward(model, state, 3, 0.5)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(initial_rng)
            actual = compiled(state, steps=3, fire_rate=0.5)
        self.assert_exact(actual, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_compiled_fc0_child_uses_fallback_and_matches_original(self):
        model = self.make_model().eval()
        state = self.make_state()
        reference = deepcopy(model)
        model.fc0 = torch.compile(model.fc0, backend="eager", fullgraph=True)
        self.addCleanup(torch.compiler.reset)
        # Do not replace torch.nn.functional.relu globally while Dynamo is
        # discovering PyTorch operators. Spy only on NCA's functional binding.
        relu = Mock(wraps=_ORIGINAL_RELU)
        functional = SimpleNamespace(
            conv2d=F.conv2d, max_pool2d=F.max_pool2d, relu=relu
        )
        with torch.inference_mode():
            initial_rng = torch.get_rng_state()
            expected = original_forward(reference, state, 3, 0.5)
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(initial_rng)
            with patch("nca.F", functional):
                actual = model(state, steps=3, fire_rate=0.5)
        self.assert_relu_mode(relu, False, calls=3)
        self.assert_exact(actual, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_non_cpu_meta_device_uses_fallback(self):
        # Branch coverage only; this does not claim CUDA/MPS numeric coverage.
        model = NCA(channels=8, hidden_size=12).to("meta").eval()
        state = torch.empty(2, 8, 7, 11, device="meta")
        with torch.inference_mode(), patch("nca.F.relu", wraps=_ORIGINAL_RELU) as relu:
            actual = model(state, fire_rate=1.0)
        self.assert_relu_mode(relu, False)
        self.assertEqual(actual.device.type, "meta")
        self.assertEqual(actual.shape, state.shape)


if __name__ == "__main__":
    unittest.main()
