"""Raster fast paths must preserve pixels, effect timing, and simulation state."""
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame
import torch

import arena
import clash
from nca import NCA


def synthetic_bundle(target, device, *args, **kwargs):
    return {"model": NCA(channels=8, hidden_size=16).to(device).eval(),
            "channels": 8, "grid_size": 16, "channels_last": False,
            "name": target.stem, "health": "ready", "seed_dir": "seed_000",
            "score": .01, "source": "synthetic-render-test"}


class RenderEfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch("arena.ensure_model", side_effect=synthetic_bundle)
        self.loader.start()
        args = arena.parse_args(["--device", "cpu", "--cpu-threads", "1", "--grid-size", "16"])
        self.world = arena.Arena(args, clash.list_targets())
        self.ui = arena.ArenaUI(self.world)
        self.ui.draw(dt=0)

    def tearDown(self):
        pygame.quit()
        self.loader.stop()

    def test_empty_and_just_expired_effects_skip_overlay_without_changing_pixels(self):
        for reduced in (False, True):
            for age in (None, 1.25, 2.0):
                with self.subTest(reduced=reduced, age=age):
                    self.ui.reduced_motion = reduced
                    self.ui.elapsed = 3.0
                    self.ui.effects = [] if age is None else [{"born": 3.0 - age}]
                    before = pygame.image.tostring(self.ui.window, "RGB")
                    with patch("pygame.Surface", wraps=pygame.Surface) as surface:
                        self.ui.draw_effects()
                    surface.assert_not_called()
                    self.assertEqual(self.ui.effects, [])
                    self.assertEqual(pygame.image.tostring(self.ui.window, "RGB"), before)

    def test_live_damage_and_plant_effects_keep_motion_and_expire_while_paused(self):
        for reduced in (False, True):
            with self.subTest(reduced=reduced):
                self.ui.action("reset")
                self.ui.reduced_motion = reduced
                self.ui.paused = True
                self.ui.interact(5, 6)
                self.ui.interact(10, 11, 0)
                self.ui.elapsed += .25
                with patch("pygame.Surface", wraps=pygame.Surface) as surface, \
                        patch("pygame.draw.circle", wraps=pygame.draw.circle) as circle, \
                        patch("pygame.draw.line", wraps=pygame.draw.line) as line:
                    self.ui.draw_effects()
                surface.assert_called_once_with(self.ui.board.size, pygame.SRCALPHA)
                self.assertEqual(circle.call_count, 2)
                self.assertEqual(line.call_count, 2 if reduced else 10)
                scale = self.ui.board.width / self.world.size
                growth = 0 if reduced else .25 / 1.25 * 16
                self.assertEqual(circle.call_args_list[0].args[3], round(self.ui.radius * scale + growth))
                self.assertEqual(circle.call_args_list[1].args[3], round(scale + growth))
                for _ in range(5):
                    self.ui.draw(dt=.25)
                self.assertEqual(self.ui.effects, [])
                self.assertEqual(self.world.steps, 0)

    def test_thumbnails_scale_once_and_match_uncached_compositing_after_status_changes(self):
        with patch("pygame.transform.smoothscale", wraps=pygame.transform.smoothscale) as scale:
            self.ui = arena.ArenaUI(self.world)
        self.assertEqual(scale.call_count, len(self.world.targets))
        self.assertTrue(all(thumb.get_size() == (28, 28) for thumb in self.ui.thumbs))
        originals = [pygame.image.load(str(target)).convert_alpha() for target in self.world.targets]
        for width, height in ((1100, 902), (860, 740)):
            for ready in (False, True, False):
                with self.subTest(window=(width, height), ready=ready):
                    for status in self.ui.status:
                        status["status"] = "ready" if ready else "collapsed"
                    with patch("pygame.transform.smoothscale", wraps=pygame.transform.smoothscale) as scale, \
                            patch("pygame.mouse.get_pos", return_value=(-1, -1)):
                        self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=width, h=height))
                        self.ui.draw(dt=0)
                    scale.assert_not_called()
                    for i, original in enumerate(originals):
                        tile = next(rect for rect, action in self.ui.buttons if action == ("target", i))
                        expected = pygame.Surface((28, 28))
                        expected.fill((53, 63, 61) if i == self.world.left_index else (27, 33, 35))
                        thumb = pygame.transform.smoothscale(original, (28, 28))
                        if not ready:
                            thumb.set_alpha(60)
                        expected.blit(thumb, (0, 0))
                        actual = self.ui.window.subsurface((tile.x + 29, tile.y + 2, 28, 28))
                        self.assertEqual(pygame.image.tostring(actual, "RGB"),
                                         pygame.image.tostring(expected, "RGB"))

    def test_fast_paths_preserve_tensors_and_all_rngs(self):
        self.world.step(2)
        self.ui.interact(4, 4, 0)
        tensors = [value.clone() for value in (self.world.a, self.world.b, self.world.owner, self.world.control)]
        torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
        for reduced in (False, True):
            self.ui.reduced_motion = reduced
            for _ in range(6):
                self.ui.draw(dt=.25)
        for value, expected in zip((self.world.a, self.world.b, self.world.owner, self.world.control), tensors):
            self.assertTrue(torch.equal(value, expected))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        actual_numpy = np.random.get_state()
        self.assertEqual(actual_numpy[0], numpy_rng[0])
        np.testing.assert_array_equal(actual_numpy[1], numpy_rng[1])
        self.assertEqual(actual_numpy[2:], numpy_rng[2:])
        self.assertEqual(random.getstate(), python_rng)


if __name__ == "__main__":
    unittest.main()
