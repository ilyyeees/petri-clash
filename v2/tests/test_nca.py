"""Reference invariants protecting pretrained NCA math and weight layout."""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nca import NCA, make_seed


class NCATests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        self.rng = torch.get_rng_state()
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def tearDown(self):
        torch.set_num_threads(self.threads)
        torch.set_rng_state(self.rng)

    def test_perception_preserves_per_channel_filters_and_order(self):
        model = NCA(channels=8, hidden_size=12)
        # A checkpoint may contain distinct filters per channel. Optimizations
        # must not quietly assume these buffers are still repeated copies.
        model.perception_filters.copy_(torch.randn_like(model.perception_filters))
        state = torch.randn(2, 8, 7, 9)
        expected = torch.cat([
            F.conv2d(state[:, channel:channel + 1], model.perception_filters[channel * 3:channel * 3 + 3], padding=1)
            for channel in range(model.channels)
        ], dim=1)
        torch.testing.assert_close(model.perceive(state), expected)

    def test_perception_identity_channel(self):
        model = NCA(channels=8, hidden_size=12)
        state = torch.randn(2, 8, 7, 9)
        torch.testing.assert_close(model.perceive(state)[:, ::3], state, rtol=0, atol=0)

    def test_state_dict_layout_unchanged(self):
        model = NCA(channels=8, hidden_size=12)
        saved = model.state_dict()
        self.assertEqual(set(saved), {"perception_filters", "fc0.weight", "fc0.bias", "fc1.weight", "fc1.bias"})
        loaded = NCA(channels=8, hidden_size=12)
        loaded.load_state_dict(saved, strict=True)
        state = torch.randn(1, 8, 7, 9)
        torch.testing.assert_close(model(state, fire_rate=1), loaded(state, fire_rate=1), rtol=0, atol=0)

    def test_seed_has_exactly_alpha_and_hidden_pulses(self):
        seed = make_seed(2, channels=8, height=7, width=9, xs=[1, 7], ys=[2, 5], device="cpu")
        self.assertEqual(tuple(seed.shape), (2, 8, 7, 9))
        self.assertEqual(int(torch.count_nonzero(seed)), 4)
        for batch, x, y in ((0, 1, 2), (1, 7, 5)):
            self.assertEqual(seed[batch, 3, y, x].item(), 1)
            self.assertEqual(seed[batch, 4, y, x].item(), 1)

    def test_seeded_async_updates_replay(self):
        model = NCA(channels=8, hidden_size=12)
        torch.nn.init.normal_(model.fc1.weight, std=0.01)
        state = make_seed(1, channels=8, height=9)
        torch.manual_seed(19)
        first = model(state, steps=4)
        torch.manual_seed(19)
        second = model(state, steps=4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_autograd_still_reaches_model_parameters(self):
        model = NCA(channels=8, hidden_size=12)
        torch.nn.init.normal_(model.fc1.weight, std=0.01)
        state = make_seed(1, channels=8, height=9).requires_grad_()
        model(state, fire_rate=1).square().sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
