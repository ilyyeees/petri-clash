"""Battle regressions. Run: python -m unittest discover -s v2/tests -v."""

import os
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from battle import clash_step, compose_rgba, crater, momentum_owner


class IdentityModel:
    def __call__(self, state, steps=1):
        return state.clone()


class EmptyModel:
    def __call__(self, state, steps=1):
        return torch.zeros_like(state)


class DelayedGrowth:
    """A frontier needs three retained hidden updates before alpha appears."""
    def __call__(self, state, steps=1):
        result = state.clone()
        near_life = F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > 0.1
        result[:, 4:5] += near_life.to(state.dtype)
        mature = result[:, 4:5] >= 3
        result[:, 3:4] = torch.where(mature, torch.ones_like(result[:, 3:4]), result[:, 3:4])
        return result


class BattleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def world(self, size=13, channels_a=6, channels_b=8):
        a = torch.zeros(1, channels_a, size, size)
        b = torch.zeros(1, channels_b, size, size)
        owner = torch.zeros(1, 1, size, size, dtype=torch.long)
        control = torch.zeros(1, 1, size, size)
        return a, b, owner, control

    def seed(self, world, side=1, x=6, y=6):
        a, b, owner, control = world
        state = a if side == 1 else b
        state[0, 3:, y, x] = 1
        owner[0, 0, y, x] = side
        control[0, 0, y, x] = 1 if side == 1 else -1
        return world

    def test_delayed_frontier_keeps_hidden_state_before_capture(self):
        world = self.seed(self.world())
        world = clash_step(*world, DelayedGrowth(), EmptyModel())
        self.assertEqual(world[0][0, 4, 6, 7], 1)
        self.assertEqual(world[2][0, 0, 6, 7], 0)
        self.assertEqual(world[0][0, 3, 6, 7], 0)
        for _ in range(5):
            world = clash_step(*world, DelayedGrowth(), EmptyModel())
        self.assertEqual(world[2][0, 0, 6, 7], 1)
        self.assertGreater(int((world[0][:, 3:4] > 0.1).sum()), 9)

    def test_growth_can_advance_faster_than_ownership(self):
        world = self.seed(self.world())
        for _ in range(10):
            world = clash_step(*world, DelayedGrowth(), EmptyModel(), pressure_gain=0.01)
        self.assertGreater(int((world[0][:, 3:4] > 0.1).sum()), 9)
        self.assertEqual(int((world[2] == 1).sum()), 1)

    def test_opposing_owned_territory_suppresses_latent_state(self):
        world = self.seed(self.seed(self.world(), x=5), side=2, x=6)
        world[0][0, :, 6, 6] = 9  # stale enemy hidden state must be removed before inference
        for _ in range(4):
            world = clash_step(*world, IdentityModel(), IdentityModel())
            self.assertEqual(int(world[2][0, 0, 6, 6]), 2)
            self.assertEqual(float(world[0][0, :, 6, 6].abs().sum()), 0)
            self.assertEqual(float(world[1][0, :, 6, 5].abs().sum()), 0)

    def test_stronger_neighbor_can_release_and_capture_enemy_territory(self):
        class Invader:
            def __call__(self, state, steps=1):
                result = state.clone()
                supported = F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > 0.1
                result[:, 3:4] = torch.where(supported, torch.full_like(result[:, 3:4], 0.9), 0.0)
                return result
        world = self.seed(self.seed(self.world(), x=5), side=2, x=6)
        world[1][0, 3, 6, 6] = 0.2
        first = clash_step(*world, Invader(), IdentityModel())
        self.assertEqual(int(first[2][0, 0, 6, 6]), 2)
        self.assertEqual(float(first[0][0, :, 6, 6].abs().sum()), 0)
        owners = []
        for _ in range(15):
            world = clash_step(*world, Invader(), IdentityModel())
            owners.append(int(world[2][0, 0, 6, 6]))
        self.assertIn(0, owners)  # release before capture, rather than an instant swap
        self.assertEqual(owners[-1], 1)
        self.assertEqual(float(world[1][0, :, 6, 6].abs().sum()), 0)

    def test_neutral_contested_tie_has_no_left_advantage(self):
        a, b, owner, control = self.world(channels_b=6)
        a[0, 3:, 6, 6] = b[0, 3:, 6, 6] = 1
        for _ in range(12):
            a, b, owner, control = clash_step(a, b, owner, control, IdentityModel(), IdentityModel())
        torch.testing.assert_close(a, b)
        self.assertEqual(int(owner.sum()), 0)
        self.assertEqual(float(control.abs().sum()), 0)
        self.assertEqual(float(a[0, 3, 6, 6]), 1)

    def test_swapping_teams_preserves_dynamics(self):
        torch.manual_seed(31)
        a, b, owner, control = self.world(channels_b=6)
        a.uniform_(-0.2, 1.0)
        b.uniform_(-0.2, 1.0)
        owner.random_(0, 3)
        control.uniform_(-1, 1)
        swap_owner = torch.where(owner > 0, 3 - owner, owner)
        result = clash_step(a, b, owner, control, IdentityModel(), IdentityModel())
        swapped = clash_step(b, a, swap_owner, -control, IdentityModel(), IdentityModel())
        torch.testing.assert_close(result[0], swapped[1])
        torch.testing.assert_close(result[1], swapped[0])
        torch.testing.assert_close(result[3], -swapped[3])
        torch.testing.assert_close(result[2], torch.where(swapped[2] > 0, 3 - swapped[2], swapped[2]))

    def test_hysteresis_retains_incumbent_until_release(self):
        owner = torch.tensor([1, 1, 2, 2, 0, 0])
        control = torch.tensor([0.3, 0.1, -0.3, -0.1, 0.54, -0.54])
        self.assertEqual(momentum_owner(control, owner, 0.55, 0.12).tolist(), [1, 0, 2, 0, 0, 0])

    def test_control_crossing_zero_cannot_leave_wrong_owner(self):
        owner = torch.tensor([1, 2, 1, 2])
        control = torch.tensor([-0.4, 0.4, -0.8, 0.8])
        self.assertEqual(momentum_owner(control, owner, 0.55, 0.12).tolist(), [0, 0, 2, 1])

    def test_dead_territory_and_stale_neutral_momentum_are_pruned(self):
        world = self.seed(self.world())
        world[3][0, 0, 0, 0] = 0.3
        result = clash_step(*world, EmptyModel(), EmptyModel())
        self.assertTrue(all(not tensor.any() for tensor in result))

    def test_distant_spontaneous_proposals_do_not_teleport(self):
        class EverywhereModel:
            def __call__(self, state, steps=1):
                return torch.ones_like(state)
        world = self.seed(self.world())
        result = clash_step(*world, EverywhereModel(), EmptyModel(), pressure_gain=1)
        self.assertEqual(int((result[0][:, 3:4] > 0.1).sum()), 9)
        self.assertEqual(int((result[2] == 1).sum()), 9)

    def test_nonfinite_cells_are_quarantined(self):
        class UnstableModel:
            def __call__(self, state, steps=1):
                state = state.clone()
                state[:, 4, 6, 7] = float("inf")
                state[:, 4, 6, 5] = float("nan")
                return state
        world = self.seed(self.world())
        world[3][0, 0, 6, 6] = float("nan")
        result = clash_step(*world, UnstableModel(), EmptyModel())
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in result))
        self.assertEqual(float(result[0][0, :, 6, 7].abs().sum()), 0)

    def test_many_ticks_are_finite_and_control_stays_bounded(self):
        world = self.seed(self.seed(self.world(), x=3), side=2, x=9)
        for _ in range(100):
            world = clash_step(*world, DelayedGrowth(), DelayedGrowth(), pressure_gain=2)
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in world))
        self.assertLessEqual(float(world[3].abs().max()), 1)
        self.assertTrue(((world[2] >= 0) & (world[2] <= 2)).all())

    def test_crater_erases_exact_circle_and_preserves_inputs(self):
        world = self.world()
        for tensor in world:
            tensor.fill_(1)
        result = crater(*world, 6, 6, 2)
        yy, xx = torch.meshgrid(torch.arange(13), torch.arange(13), indexing="ij")
        mask = (xx - 6).square() + (yy - 6).square() <= 4
        self.assertEqual(int(mask.sum()), 13)
        for before, after in zip(world, result):
            self.assertTrue((before == 1).all())
            self.assertTrue((after[..., mask] == 0).all())
            self.assertTrue((after[..., ~mask] == 1).all())

    def test_edge_crater_does_not_wrap_or_clamp_center(self):
        world = self.world()
        for tensor in world:
            tensor.fill_(1)
        result = crater(*world, -1, 0, 1)
        self.assertEqual(int((result[2] == 0).sum()), 1)
        self.assertEqual(int(result[2][0, 0, 0, 0]), 0)
        self.assertEqual(int(result[2][0, 0, 0, -1]), 1)
        self.assertTrue(torch.equal(crater(*world, -100, -100, 1)[2], world[2]))

    def test_crater_zero_radius_and_oversized_radius(self):
        world = self.seed(self.world())
        self.assertTrue(all(not tensor.any() for tensor in crater(*world, 6, 6, 0)))
        self.assertTrue(all(not tensor.any() for tensor in crater(*world, 0, 0, 100)))

    def test_crater_removes_frontier_hidden_state_as_well(self):
        world = clash_step(*self.seed(self.world()), DelayedGrowth(), EmptyModel())
        self.assertGreater(float(world[0][0, 4, 6, 7]), 0)
        damaged = crater(*world, 7, 6, 0)
        self.assertEqual(float(damaged[0][0, :, 6, 7].abs().sum()), 0)
        self.assertEqual(float(damaged[3][0, 0, 6, 7]), 0)

    def test_step_does_not_mutate_inputs(self):
        world = self.seed(self.world())
        copies = tuple(tensor.clone() for tensor in world)
        clash_step(*world, DelayedGrowth(), EmptyModel())
        for tensor, copy in zip(world, copies):
            torch.testing.assert_close(tensor, copy)

    def test_invalid_settings_are_rejected(self):
        for settings in ({"pressure_gain": -1}, {"control_decay": 1.1}, {"capture_threshold": 0.1},
                         {"release_threshold": -1}, {"tie_margin": float("nan")}, {"capture_threshold": 2}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                clash_step(*self.world(), IdentityModel(), IdentityModel(), **settings)
        for radius in (-1, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                crater(*self.world(), 6, 6, radius)

    def test_invalid_shapes_are_rejected(self):
        a, b, owner, control = self.world()
        with self.assertRaises(ValueError):
            clash_step(a, b[:, :, :-1], owner, control, IdentityModel(), IdentityModel())
        with self.assertRaises(ValueError):
            clash_step(a[:, :3], b, owner, control, IdentityModel(), IdentityModel())

    def test_neutral_rgba_is_visible_and_team_symmetric(self):
        a, b, owner, control = self.world(channels_b=6)
        a[:, 0], a[:, 3] = 1, 0.5
        b[:, 2], b[:, 3] = 1, 0.5
        rgba = compose_rgba(a, b, owner)
        torch.testing.assert_close(rgba, compose_rgba(b, a, owner))
        torch.testing.assert_close(rgba[0, :, 6, 6], torch.tensor([0.5, 0.0, 0.5, 0.75]))
        self.assertFalse(compose_rgba(a, b, owner, show_frontier=False).any())

    def test_owned_rgba_excludes_opponent(self):
        a, b, owner, control = self.world(channels_b=6)
        a[:, 0], a[:, 3] = 1, 1
        b[:, 2], b[:, 3] = 1, 1
        owner.fill_(1)
        torch.testing.assert_close(compose_rgba(a, b, owner), a[:, :4])
        owner.fill_(2)
        torch.testing.assert_close(compose_rgba(a, b, owner), b[:, :4])


@unittest.skipUnless(os.environ.get("PETRI_TEST_CHECKPOINTS") == "1", "set PETRI_TEST_CHECKPOINTS=1 for pretrained CPU smoke tests")
class PretrainedBattleTests(unittest.TestCase):
    """Optional checks use the shipped accepted checkpoints without training."""

    def test_accepted_organisms_retain_near_solo_growth(self):
        from nca import NCA, make_seed
        torch.set_num_threads(1)
        for target, seed in (("01_heart", 0), ("02_star", 1), ("03_sun", 2), ("06_flower", 2), ("07_umbrella", 0)):
            with self.subTest(target=target, seed=seed):
                path = Path(__file__).resolve().parents[1] / "weights" / target / f"seed_{seed:03d}" / "checkpoints" / "best.pt"
                self.assertTrue(path.is_file(), f"missing checkpoint: {path}")
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                config = checkpoint["config"]
                model = NCA(**{key: config["model"][key] for key in ("channels", "hidden_size", "fire_rate")})
                model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()})
                model.eval()
                channels, size = config["model"]["channels"], config["data"]["grid_size"]
                counts = []
                for hard in (False, True):
                    torch.manual_seed(42)
                    a = make_seed(1, channels=channels, height=size)
                    b = torch.zeros_like(a)
                    owner = torch.zeros(1, 1, size, size, dtype=torch.long)
                    owner[0, 0, size // 2, size // 2] = 1
                    control = owner.float()
                    with torch.inference_mode():
                        for _ in range(128):
                            if hard:
                                a, b, owner, control = clash_step(a, b, owner, control, model, EmptyModel())
                            else:
                                a = model(a, steps=1)
                    self.assertTrue(torch.isfinite(a).all())
                    counts.append(int((a[:, 3:4] > 0.1).sum()))
                self.assertGreater(counts[1], 60)
                self.assertGreaterEqual(counts[1], counts[0] * 0.85)


if __name__ == "__main__":
    unittest.main()
