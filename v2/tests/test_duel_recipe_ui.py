"""CPU/SDL contracts for saving a live duel as a portable replay recipe.

Small deterministic model bundles keep these tests focused on integration,
input routing, state preservation and honest export provenance. They do not
measure trained-model quality or make cross-machine replay guarantees.
"""

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
import textwrap
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

import numpy as np
import pygame
import torch

import arena
import clash
from nca import NCA


def synthetic_bundle(target, device, bootstrap_steps=0, preferred_seed=None,
                     allow_unhealthy=False):
    """Pin a real-looking resolved seed independently of launch defaults."""
    seed = preferred_seed if preferred_seed is not None else int(target.stem[:2]) + 2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(sum(map(ord, target.stem)) + seed)
        model = NCA(channels=8, hidden_size=16).to(device).eval()
        with torch.no_grad():
            model.fc1.weight.normal_(std=.015)
            model.fc1.bias[3] = .06
    return {"model": model, "channels": 8, "grid_size": 16,
            "channels_last": False, "name": target.stem,
            "health": "ready", "seed_dir": f"seed_{seed:03d}", "score": .01,
            "kind": "synthetic-untrained",
            "source": f"/private/duel-recipe-test/{target.stem}/seed_{seed:03d}/best.pt"}


class RecipeTestCase(unittest.TestCase):
    def setUp(self):
        self.torch_rng = torch.get_rng_state()
        self.numpy_rng = np.random.get_state()
        self.python_rng = random.getstate()
        self.threads = torch.get_num_threads()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.save_dir = self.folder / "saved-duels"
        self.loader = patch("arena.ensure_model", side_effect=synthetic_bundle)
        self.load = self.loader.start()
        self.new_world()

    def tearDown(self):
        pygame.quit()
        self.loader.stop()
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(self.deterministic)
        torch.set_rng_state(self.torch_rng)
        np.random.set_state(self.numpy_rng)
        random.setstate(self.python_rng)

    def arguments(self, *extra, duel=True):
        return ["--device", "cpu", "--cpu-threads", "1", "--grid-size", "16",
                "--round-ticks", "9", "--warmup-ticks", "2", "--seed", "27",
                "--duel-save-dir", str(self.save_dir),
                *(["--duel"] if duel else []), *extra]

    def new_world(self, *extra, duel=True):
        args = arena.parse_args(self.arguments(*extra, duel=duel))
        self.world = arena.Arena(args, clash.list_targets())
        return self.world

    def finish(self):
        self.world.step(self.world.duel_config.total_ticks + 8)
        self.assertEqual(self.world.duel.phase, "finished")

    def random_states(self):
        return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()

    def assert_rng_equal(self, expected):
        actual = self.random_states()
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertEqual(actual[1][0], expected[1][0])
        np.testing.assert_array_equal(actual[1][1], expected[1][1])
        self.assertEqual(actual[1][2:], expected[1][2:])
        self.assertEqual(actual[2], expected[2])

    def tensor_bytes(self):
        return tuple((value.dtype, tuple(value.shape), value.detach().cpu().numpy().tobytes())
                     for value in (self.world.a, self.world.b,
                                   self.world.owner, self.world.control))

    def frozen_state(self):
        return self.world.duel, self.world.steps, self.tensor_bytes(), self.world.duel.snapshot()

    def assert_frozen(self, expected, rng):
        self.assertIs(self.world.duel, expected[0])
        self.assertEqual(self.world.steps, expected[1])
        self.assertEqual(self.tensor_bytes(), expected[2])
        self.assertEqual(self.world.duel.snapshot(), expected[3])
        self.assert_rng_equal(rng)

    def saved_files(self):
        return sorted(self.save_dir.glob("*.json"))


class DuelRecipeArenaIntegrationTests(RecipeTestCase):
    def test_export_captures_actual_rules_positions_runtime_and_integer_scores(self):
        for mode in ("hard", "soft"):
            for custom in (False, True):
                with self.subTest(mode=mode, custom=custom):
                    positions = ["--left-pos=-50,80", "--right-pos=80,-50"] if custom else []
                    self.new_world("--mode", mode, "--pressure-gain", ".31",
                                   "--control-decay", ".89", "--capture-threshold", ".61",
                                   "--release-threshold", ".13", "--tie-margin", ".03",
                                   *positions)
                    self.finish()
                    recipe = self.world.duel_recipe()
                    config = recipe["configuration"]
                    report = self.world.report(0)
                    self.assertEqual(config["mode"], mode)
                    self.assertEqual(config["simulation_seed"], 27)
                    self.assertEqual(config["grid_size"], 16)
                    self.assertEqual(config["round_ticks"], 9)
                    self.assertEqual(config["warmup_ticks"], 2)
                    self.assertEqual(config["starting_positions"], report["starting_positions"])
                    self.assertEqual(config["placement"], "custom" if custom else "mirrored")
                    self.assertEqual(config["rules"], report["rules"])
                    if custom:
                        self.assertEqual(config["starting_positions"],
                                         {"left": [2, 13], "right": [13, 2]})
                    self.assertEqual(recipe["runtime"]["device"], "cpu")
                    self.assertEqual(recipe["runtime"]["cpu_threads"], 1)
                    self.assertEqual(recipe["runtime"]["deterministic_algorithms"],
                                     torch.are_deterministic_algorithms_enabled())
                    for side in ("left", "right"):
                        score = recipe["expected_score"][f"{side}_cell_ticks"]
                        self.assertIs(type(score), int)
                        self.assertEqual(score, report["duel"]["scores"][side])
                    text = json.dumps(recipe, allow_nan=False)
                    self.assertNotIn("/private/", text)
                    self.assertNotIn("source", recipe["models"]["left"])
                    # The established diagnostic report intentionally retains its source.
                    self.assertIn("/private/", report["left"]["source"])

    def test_export_uses_loaded_cultures_and_resolved_seeds_after_selection(self):
        self.new_world("--grid-size", "0")
        self.assertIsNone(self.world.args.left_seed)
        self.assertIsNone(self.world.args.right_seed)
        self.world.select(5, 0)
        self.world.select(6, 1)
        self.world.mode = "soft"
        self.world.reset()
        self.finish()
        self.assertEqual((self.world.args.left, self.world.args.right, self.world.args.mode),
                         (1, 2, "hard"))
        recipe = self.world.duel_recipe()
        self.assertEqual(recipe["configuration"]["mode"], "soft")
        self.assertEqual(self.world.args.grid_size, 0)
        self.assertEqual(recipe["configuration"]["grid_size"], 16)
        self.assertEqual(recipe["models"], {
            "left": {"target": "06_flower", "checkpoint_seed": 8},
            "right": {"target": "07_umbrella", "checkpoint_seed": 9}})

    def test_explicit_checkpoint_seed_zero_and_nonzero_are_preserved(self):
        self.new_world("--left-seed", "0", "--right-seed", "23")
        self.finish()
        recipe = self.world.duel_recipe()
        self.assertEqual(recipe["models"]["left"]["checkpoint_seed"], 0)
        self.assertEqual(recipe["models"]["right"]["checkpoint_seed"], 23)
        self.assertEqual(self.load.call_args.kwargs["preferred_seed"], 23)

    def test_runtime_records_current_thread_count_rather_than_launch_argument(self):
        self.finish()
        self.assertEqual(self.world.args.cpu_threads, 1)
        torch.set_num_threads(2)
        frozen, rng = self.frozen_state(), self.random_states()
        self.assertEqual(self.world.duel_recipe()["runtime"]["cpu_threads"], 2)
        self.assert_frozen(frozen, rng)

    def test_repeated_export_is_pure_and_returned_nested_values_are_independent(self):
        self.finish()
        frozen, rng = self.frozen_state(), self.random_states()
        original_report = self.world.report(0)
        original_args = copy.deepcopy(vars(self.world.args))
        first = self.world.duel_recipe()
        expected = copy.deepcopy(first)
        for _ in range(4):
            self.assertEqual(self.world.duel_recipe(), expected)
        first["configuration"]["starting_positions"]["left"][0] = 99
        first["configuration"]["rules"]["tie_margin"] = .8
        first["models"]["left"]["target"] = "changed"
        first["expected_score"]["left_cell_ticks"] = 0
        self.assertEqual(self.world.duel_recipe(), expected)
        self.assertEqual(self.world.report(0), original_report)
        self.assertEqual(vars(self.world.args), original_args)
        self.assert_frozen(frozen, rng)

    def test_each_nonfinite_state_tensor_prevents_export_without_rewriting_result(self):
        self.finish()
        for name in ("a", "b", "owner", "control"):
            for value in (float("nan"), float("inf")):
                with self.subTest(tensor=name, nonfinite=value):
                    original = getattr(self.world, name)
                    changed = original.float().clone()
                    changed.reshape(-1)[0] = value
                    setattr(self.world, name, changed)
                    frozen, rng = self.frozen_state(), self.random_states()
                    with self.assertRaises(ValueError):
                        self.world.duel_recipe()
                    self.assert_frozen(frozen, rng)
                    setattr(self.world, name, original)
        self.assertIsInstance(self.world.duel_recipe(), dict)

    def test_sandbox_lesson_partial_and_invalid_rounds_have_no_recipe(self):
        for scenario in ("sandbox", "lesson", "partial", "invalid"):
            with self.subTest(scenario=scenario):
                self.new_world(duel=scenario not in ("sandbox", "lesson"))
                if scenario == "lesson":
                    self.world.start_lesson()
                elif scenario == "partial":
                    self.world.step(3)
                elif scenario == "invalid":
                    with torch.no_grad():
                        self.world.left["model"].fc1.bias[3] = float("nan")
                    self.world.step(20)
                    self.assertEqual(self.world.duel.phase, "invalid")
                tensors, rng = self.tensor_bytes(), self.random_states()
                with self.assertRaises(ValueError):
                    self.world.duel_recipe()
                self.assertEqual(self.tensor_bytes(), tensors)
                self.assert_rng_equal(rng)

    def test_unready_bundles_and_non_cpu_resolved_devices_cannot_export(self):
        self.finish()
        frozen, rng = self.frozen_state(), self.random_states()
        for side in ("left", "right"):
            bundle = getattr(self.world, side)
            for health in ("collapsed", "unverified", None):
                with self.subTest(side=side, health=health):
                    bundle["health"] = health
                    with self.assertRaises(ValueError):
                        self.world.duel_recipe()
                    self.assert_frozen(frozen, rng)
            bundle["health"] = "ready"
        for device in ("cuda", "mps"):
            with self.subTest(device=device):
                # No accelerator allocation is needed to test export's device gate.
                self.world.args.device = device
                with self.assertRaises(ValueError):
                    self.world.duel_recipe()
                self.assert_frozen(frozen, rng)
        self.world.args.device = "cpu"

    def test_each_off_cpu_state_tensor_is_rejected_before_report_or_inspection(self):
        self.finish()
        for name in ("a", "b", "owner", "control"):
            with self.subTest(tensor=name):
                original = getattr(self.world, name)
                # The meta device checks the CPU-only gate without CUDA/MPS hardware.
                setattr(self.world, name, torch.empty_like(original, device="meta"))
                rng = self.random_states()
                with patch.object(self.world, "report", side_effect=AssertionError("inspected unsupported state")):
                    with self.assertRaisesRegex(ValueError, "CPU"):
                        self.world.duel_recipe()
                self.assert_rng_equal(rng)
                setattr(self.world, name, original)


class DuelRecipeUIIntegrationTests(RecipeTestCase):
    def new_world(self, *extra, duel=True):
        super().new_world(*extra, duel=duel)
        self.ui = arena.ArenaUI(self.world)
        self.ui.draw(dt=0)
        return self.world

    def finish(self):
        super().finish()
        self.ui.draw(dt=0)
        self.assertTrue(self.ui.paused)

    def key(self, key):
        self.ui.event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))

    def click(self, position):
        self.ui.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=position, button=1))

    def result_control(self, action):
        return next(rect for rect, value in self.ui.result_buttons if value == action)

    def test_save_label_and_hit_geometry_fit_normal_and_minimum_window(self):
        self.finish()
        for size in ((1100, 902), (860, 740), (200, 200)):
            with self.subTest(size=size):
                with patch.object(self.ui, "button", wraps=self.ui.button) as buttons, \
                        patch.object(self.ui, "text", wraps=self.ui.text) as text:
                    self.ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=size[0], h=size[1]))
                save = self.result_control("save_duel")
                labels = [call.args[1] for call in buttons.call_args_list
                          if call.args[2] == "save_duel"]
                self.assertEqual(len(labels), 1)
                self.assertIn("SAVE DUEL", labels[0])
                self.assertTrue(labels[0].endswith("/ S"))
                self.assertLessEqual(self.ui.button_font.size(labels[0])[0], save.width - 4)
                self.assertTrue(self.ui.result_rect.contains(save))
                self.assertTrue(self.ui.board.contains(save))
                self.assertTrue(self.ui.window.get_rect().contains(self.ui.result_rect))
                for index, (rect, _) in enumerate(self.ui.result_buttons):
                    for other, _ in self.ui.result_buttons[index + 1:]:
                        self.assertFalse(rect.colliderect(other))
                for call in text.call_args_list:
                    if call.args[0] == "SEEDED DUEL / FINAL":
                        _, x, y, font_size, *_ = call.args
                        bounds = pygame.Rect((x, y), self.ui.fonts[font_size].size(call.args[0]))
                        self.assertFalse(save.colliderect(bounds))

    def test_queued_saves_write_once_without_advancing_or_editing_the_board(self):
        self.finish()
        frozen, rng = self.frozen_state(), self.random_states()
        position = self.result_control("save_duel").center
        with patch.object(self.ui, "interact", wraps=self.ui.interact) as interact, \
                patch.object(self.world, "damage", wraps=self.world.damage) as damage:
            self.click(position)
            self.click(position)
            self.key(pygame.K_s)
            self.key(pygame.K_s)
            self.ui.draw(dt=0)
            self.click(self.result_control("save_duel").center)
        interact.assert_not_called()
        damage.assert_not_called()
        self.assertEqual(len(self.saved_files()), 1)
        self.assertEqual(json.loads(self.saved_files()[0].read_text()), self.world.duel_recipe())
        self.assertIs(self.ui.saved_duel, self.world.duel)
        self.assertEqual(self.ui.saved_duel_path, self.saved_files()[0])
        self.assertTrue(self.ui.paused)
        self.assertIn("saved", self.ui.message.lower())
        self.assert_frozen(frozen, rng)

    def test_keyboard_save_works_while_result_is_hidden_and_stays_hidden(self):
        self.finish()
        self.ui.action("result")
        self.ui.draw(dt=0)
        frozen, rng = self.frozen_state(), self.random_states()
        self.assertFalse(self.ui.result_visible)
        self.assertEqual(self.ui.result_buttons, [])
        self.key(pygame.K_s)
        self.key(pygame.K_s)
        self.ui.draw(dt=0)
        self.assertEqual(len(self.saved_files()), 1)
        self.assertFalse(self.ui.result_visible)
        self.assertIsNone(self.ui.result_rect)
        self.assert_frozen(frozen, rng)

    def test_each_fresh_round_can_save_including_same_seed_rematches(self):
        self.finish()
        self.key(pygame.K_s)
        previous = self.world.duel
        previous_contents = {path: path.read_bytes() for path in self.saved_files()}
        for action in ("reset", "next_seed", "mode", ("target", 5)):
            with self.subTest(action=action):
                self.ui.action(action)
                self.assertIsNot(self.world.duel, previous)
                self.assertFalse(self.world.duel.finished)
                self.assertEqual(self.world.steps, 0)
                self.key(pygame.K_s)
                self.assertEqual(len(self.saved_files()), len(previous_contents))
                self.finish()
                self.key(pygame.K_s)
                self.assertEqual(len(self.saved_files()), len(previous_contents) + 1)
                for path, contents in previous_contents.items():
                    self.assertEqual(path.read_bytes(), contents)
                recipe = json.loads(self.ui.saved_duel_path.read_text())
                self.assertEqual(recipe, self.world.duel_recipe())
                self.assertEqual(recipe["configuration"]["mode"], self.world.mode)
                self.assertEqual(recipe["models"]["left"]["target"], self.world.left["name"])
                previous = self.world.duel
                previous_contents = {path: path.read_bytes() for path in self.saved_files()}

    def test_next_seed_wrap_and_queued_click_save_only_the_new_round(self):
        self.new_world("--seed", str(2 ** 32 - 1))
        self.finish()
        self.key(pygame.K_s)
        first = json.loads(self.ui.saved_duel_path.read_text())
        position = self.result_control("next_seed").center
        self.click(position)
        self.click(position)
        self.key(pygame.K_s)
        self.assertEqual(self.world.args.seed, 0)
        self.assertEqual(self.world.steps, 0)
        self.assertEqual(len(self.saved_files()), 1)
        self.finish()
        self.key(pygame.K_s)
        second = json.loads(self.ui.saved_duel_path.read_text())
        self.assertEqual(len(self.saved_files()), 2)
        self.assertEqual(first["configuration"]["simulation_seed"], 2 ** 32 - 1)
        self.assertEqual(second["configuration"]["simulation_seed"], 0)
        self.assertEqual(first["models"], second["models"])

    def test_write_failure_leaves_frozen_result_and_randomness_intact_then_retries(self):
        self.finish()
        self.save_dir.write_text("an existing file blocks directory creation")
        frozen, rng = self.frozen_state(), self.random_states()
        self.key(pygame.K_s)
        self.assert_frozen(frozen, rng)
        self.assertTrue(self.ui.paused)
        self.assertIsNone(self.ui.saved_duel)
        self.assertIsNone(self.ui.saved_duel_path)
        self.assertIn("could not save", self.ui.message.lower())
        self.assertEqual(self.save_dir.read_text(), "an existing file blocks directory creation")
        self.save_dir.unlink()
        self.key(pygame.K_s)
        self.assertEqual(len(self.saved_files()), 1)
        self.assertIs(self.ui.saved_duel, frozen[0])
        self.assert_frozen(frozen, rng)

    def test_deleted_save_can_be_saved_again_for_the_same_round(self):
        self.finish()
        self.key(pygame.K_s)
        first = self.ui.saved_duel_path
        expected = json.loads(first.read_text())
        first.unlink()
        frozen, rng = self.frozen_state(), self.random_states()
        self.key(pygame.K_s)
        self.assertEqual(len(self.saved_files()), 1)
        self.assertTrue(self.ui.saved_duel_path.is_file())
        self.assertEqual(json.loads(self.ui.saved_duel_path.read_text()), expected)
        self.assert_frozen(frozen, rng)

    def test_saving_never_creates_files_for_ineligible_play_states(self):
        for scenario in ("partial", "sandbox", "lesson", "invalid", "unready", "cuda"):
            with self.subTest(scenario=scenario):
                self.new_world(duel=scenario not in ("sandbox", "lesson"))
                if scenario == "lesson":
                    self.ui.action("lesson")
                elif scenario == "invalid":
                    with torch.no_grad():
                        self.world.left["model"].fc1.bias[3] = float("nan")
                    self.world.step(20)
                    self.ui.draw(dt=0)
                elif scenario in ("unready", "cuda"):
                    self.finish()
                    if scenario == "unready":
                        self.world.left["health"] = "collapsed"
                    else:
                        self.world.args.device = "cuda"
                self.ui.draw(dt=0)
                if scenario in ("partial", "sandbox", "lesson", "invalid"):
                    self.assertNotIn("save_duel", [value for _, value in self.ui.result_buttons])
                tensors, rng, step = self.tensor_bytes(), self.random_states(), self.world.steps
                self.key(pygame.K_s)
                self.ui.action("save_duel")
                self.assertEqual(self.saved_files(), [])
                self.assertEqual(self.world.steps, step)
                self.assertEqual(self.tensor_bytes(), tensors)
                self.assert_rng_equal(rng)
                self.assertIn("save", self.ui.message.lower())


class DuelRecipeCLIIntegrationTests(RecipeTestCase):
    def test_parser_exposes_explicit_export_and_local_save_directory(self):
        args = arena.parse_args([])
        self.assertIsNone(args.export_duel)
        self.assertEqual(args.duel_save_dir, Path("duels"))
        args = arena.parse_args(["--export-duel", "finished.json", "--duel-save-dir", "saved"])
        self.assertEqual(args.export_duel, Path("finished.json"))
        self.assertEqual(args.duel_save_dir, Path("saved"))

    def test_output_aliases_rejected_before_model_load_without_changing_files(self):
        original = self.folder / "untouched.json"
        original.write_bytes(b"do not replace")
        symlink = self.folder / "symbolic.json"
        symlink.symlink_to(original)
        hardlink = self.folder / "hardlink.json"
        os.link(original, hardlink)
        lexical = self.folder / "." / original.name
        self.load.reset_mock()
        for first, second in (("--export-duel", "--report"),
                              ("--export-duel", "--snapshot"),
                              ("--report", "--snapshot")):
            for alias in (original, lexical, symlink, hardlink):
                with self.subTest(outputs=(first, second), alias=alias), \
                        contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as error:
                    arena.main(self.arguments(first, str(original), second, str(alias)))
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(original.read_bytes(), b"do not replace")
        self.load.assert_not_called()

    def test_interactive_exit_exports_current_cultures_mode_and_next_seed(self):
        output = self.folder / "current.json"
        report_path = self.folder / "diagnostic.json"

        def play(ui):
            ui.arena.step(100)
            ui.refresh()
            ui.action("next_seed")
            ui.side = 0
            ui.action(("target", 5))
            ui.side = 1
            ui.action(("target", 6))
            ui.action("mode")
            ui.arena.step(100)

        with patch.object(arena.ArenaUI, "run", play), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            arena.main(self.arguments("--export-duel", str(output),
                                      "--report", str(report_path)))
        recipe, report = json.loads(output.read_text()), json.loads(report_path.read_text())
        self.assertEqual(recipe["configuration"]["mode"], "soft")
        self.assertEqual(recipe["configuration"]["simulation_seed"], 28)
        self.assertEqual(recipe["models"]["left"]["target"], "06_flower")
        self.assertEqual(recipe["models"]["right"]["target"], "07_umbrella")
        self.assertEqual(report["duel"]["phase"], "finished")
        self.assertEqual(report["seed"], 28)
        self.assertIn("/private/", report["left"]["source"])
        self.assertNotIn("/private/", output.read_text())

    def test_headless_exports_and_rejections_never_import_pygame(self):
        # Every path runs in a new interpreter with Pygame import prohibited,
        # regardless of which SDL tests this process has already collected.
        script = textwrap.dedent("""
            import contextlib
            import importlib.abc
            import io
            import json
            from pathlib import Path
            import sys

            class NoPygame(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'pygame' or fullname.startswith('pygame.'):
                        raise AssertionError('Pygame imported by headless recipe export')

            sys.meta_path.insert(0, NoPygame())
            sys.path.insert(0, sys.argv[1])
            import arena
            import torch
            from nca import NCA

            output = Path(sys.argv[2])

            def bundle(target, device, *args, **kwargs):
                model = NCA(channels=8, hidden_size=16).to(device).eval()
                return {'model': model, 'channels': 8, 'grid_size': 16,
                        'channels_last': False, 'name': target.stem, 'health': 'ready',
                        'seed_dir': 'seed_007', 'score': .01,
                        'source': '/private/never-share/weights.pt'}

            def no_ui(*args, **kwargs):
                raise AssertionError('UI constructed by headless recipe export')

            arena.ensure_model = bundle
            arena.ArenaUI = no_ui
            for scenario in ('hard', 'soft', 'partial', 'invalid', 'sandbox'):
                recipe_path = output / (scenario + '-recipe.json')
                report_path = output / (scenario + '-report.json')
                args = ['--device', 'cpu', '--cpu-threads', '1', '--grid-size', '16',
                        '--round-ticks', '9', '--warmup-ticks', '2', '--fps', '1',
                        '--headless-frames', '4' if scenario == 'partial' else '100',
                        '--mode', 'soft' if scenario == 'soft' else 'hard',
                        '--report', str(report_path), '--export-duel', str(recipe_path)]
                if scenario != 'sandbox':
                    args.append('--duel')
                arena.ensure_model = bundle
                if scenario == 'invalid':
                    def invalid_bundle(*args, **kwargs):
                        result = bundle(*args, **kwargs)
                        with torch.no_grad():
                            result['model'].fc1.bias[3] = float('nan')
                        return result
                    arena.ensure_model = invalid_bundle
                failed = False
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        arena.main(args)
                    except SystemExit as error:
                        failed = bool(error.code)
                assert failed == (scenario not in ('hard', 'soft')), scenario
                assert recipe_path.exists() == (not failed), scenario
                assert report_path.is_file(), scenario
                report = json.loads(report_path.read_text())
                assert report['left']['source'] == '/private/never-share/weights.pt'
                if not failed:
                    recipe = json.loads(recipe_path.read_text())
                    assert recipe['expected_score'] == {'left_cell_ticks': 7, 'right_cell_ticks': 7}
                    assert recipe['configuration']['mode'] == scenario
                    assert recipe['models']['left']['checkpoint_seed'] == 7
                    assert '/private/' not in recipe_path.read_text()
                    assert report['step'] == 9
                assert 'pygame' not in sys.modules
        """)
        result = subprocess.run([sys.executable, "-c", script, str(V2_ROOT), str(self.folder)],
                                capture_output=True, text=True, timeout=60,
                                env={**os.environ, "SDL_VIDEODRIVER": "intentionally-unavailable"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
