import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
import torch

import arena
import clash
from nca import NCA


def bundle(target, device, *args, **kwargs):
    model = NCA(channels=8, hidden_size=16).to(device).eval()
    return {"model": model, "channels": 8, "grid_size": 16, "channels_last": False,
            "name": target.stem, "health": "ready", "seed_dir": "seed_000", "score": 0.01,
            "kind": "v2", "source": "test"}


class ArenaTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch("arena.ensure_model", side_effect=bundle)
        self.loader.start()
        self.args = arena.parse_args(["--device", "cpu", "--grid-size", "16"])
        self.world = arena.Arena(self.args, clash.list_targets())

    def tearDown(self):
        pygame.quit()
        self.loader.stop()

    def test_reset_replays_seed_and_state(self):
        self.world.step(5)
        expected = self.world.a.clone(), self.world.b.clone(), self.world.owner.clone()
        self.world.reset()
        self.world.step(5)
        for result, prior in zip((self.world.a, self.world.b, self.world.owner), expected):
            self.assertTrue(torch.equal(result, prior))

    def test_clear_plant_after_inference_and_damage(self):
        self.world.step(2)
        self.world.clear()
        self.world.plant(3, 4, 0)
        self.assertEqual(self.world.a[0, 3, 4, 3].item(), 1)
        self.world.plant(3, 4, 1)
        self.assertEqual(self.world.a[:, :, 4, 3].sum().item(), 0)
        self.world.damage(3, 4, 1)
        self.assertEqual(self.world.b[:, :, 4, 3].sum().item(), 0)
        self.assertEqual(self.world.owner[0, 0, 4, 3].item(), 0)

    def test_failed_selection_preserves_match(self):
        before = self.world.left, self.world.a.clone(), self.world.steps
        with patch.object(self.world, "load", side_effect=ValueError("collapsed")):
            with self.assertRaises(ValueError):
                self.world.select(3, 0)
        self.assertIs(self.world.left, before[0])
        self.assertTrue(torch.equal(self.world.a, before[1]))
        self.assertEqual(self.world.steps, before[2])

    def test_headless_never_constructs_ui_and_writes_finite_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report, snapshot = Path(folder) / "report.json", Path(folder) / "frame.png"
            with patch("arena.ArenaUI", side_effect=AssertionError("UI in headless")), \
                    patch("pygame.display.init", side_effect=AssertionError("SDL in headless")), \
                    patch("pygame.time.Clock", side_effect=AssertionError("FPS clock in headless")), \
                    contextlib.redirect_stdout(io.StringIO()):
                arena.main(["--device", "cpu", "--headless-frames", "5", "--fps", "1",
                            "--report", str(report), "--snapshot", str(snapshot)])
            result = json.loads(report.read_text())
            self.assertEqual(result["step"], 5)
            self.assertTrue(result["finite"])
            self.assertTrue(snapshot.is_file())

    def test_soft_mode_grows_both_states(self):
        self.world.mode = "soft"
        self.world.step(3)
        self.assertEqual(self.world.steps, 3)
        self.assertIsNone(self.world.stats()["left_territory"])
        self.assertTrue(self.world.stats()["finite"])
        self.assertEqual(self.world.rgb().shape, (16, 16, 3))

    def test_ui_repeated_actions_and_invalid_selection(self):
        ui = arena.ArenaUI(self.world)
        ui.draw()
        for _ in range(2):
            ui.action("pause")
            ui.action("step")
            self.assertTrue(ui.paused)
            ui.action("reset")
            self.assertEqual(self.world.steps, 0)
        with patch.object(self.world, "load", side_effect=ValueError("collapsed")):
            ui.action(("target", 3))
        self.assertIn("collapsed", ui.message)
        ui.action("mode")
        self.assertEqual(self.world.mode, "soft")
        ui.action("mode")
        self.assertEqual(self.world.mode, "hard")
        ui.action("colors")
        ui.action(("view", "pressure"))
        ui.draw()
        self.assertFalse(ui.event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)))

    def test_ui_coordinates_damage_and_sidebar_does_not_damage(self):
        ui = arena.ArenaUI(self.world)
        ui.draw()
        with patch.object(self.world, "damage") as damage:
            ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=ui.board.center, button=1))
            damage.assert_called_once_with(8, 8, ui.radius)
            ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(ui.window.get_width() - 10, 20), button=1))
            self.assertEqual(damage.call_count, 1)
        ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=200, h=200))
        ui.draw()
        self.assertGreaterEqual(ui.window.get_width(), 860)
        self.assertGreaterEqual(ui.window.get_height(), 740)

    def test_cli_rejects_invalid_values(self):
        for args in (["--grid-size", "-1"], ["--fps", "0"], ["--seed", "4294967296"],
                     ["--pressure-gain", "nan"], ["--tie-margin", "2"],
                     ["--cpu-threads", "0"], ["--release-threshold", ".9"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arena.parse_args(args)


class ModelSelectionTests(unittest.TestCase):
    def test_missing_failed_unverified_and_preferred_seed(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(clash, "V2_ROOT", Path(folder)):
            target = Path("heart.png")
            self.assertIsNone(clash.discover_v2_checkpoint(target))
            for seed, score in ((0, .6), (1, .03), (2, float("nan"))):
                run = Path(folder) / "weights" / "heart" / f"seed_{seed:03}"
                (run / "checkpoints").mkdir(parents=True)
                (run / "checkpoints" / "best.pt").touch()
                (run / "best_summary.json").write_text(json.dumps({"score": score}))
            chosen = clash.discover_v2_checkpoint(target)
            self.assertEqual(chosen.parents[1].name, "seed_001")
            self.assertIsNone(clash.discover_v2_checkpoint(target, preferred_seed=0))
            self.assertIsNone(clash.discover_v2_checkpoint(target, preferred_seed=99))
            self.assertIsNotNone(clash.discover_v2_checkpoint(target, preferred_seed=0, allow_unhealthy=True))
            self.assertEqual(clash.target_status(target)["status"], "ready")

    def test_no_silent_untrained_fallback(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(clash, "V2_ROOT", Path(folder)):
            with self.assertRaisesRegex(ValueError, "No usable checkpoint"):
                clash.ensure_model(Path("heart.png"), "cpu")


if __name__ == "__main__":
    unittest.main()
