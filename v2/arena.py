"""Interactive arena and genuinely render-free, reproducible simulation runner."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from battle import clash_step, compose_rgba, crater
from clash import (V2_ROOT, active_grid_size, clamp, ensure_model, list_targets,
                   parse_grid_pos, reset_world, target_status)
from runtime import configure_runtime, runtime_info
from trainer.common import pick_device
from duel import DuelConfig, DuelRound


def checkpoint_caption(bundle):
    """Short loaded identity; never substitute a catalog's best checkpoint."""
    seed = bundle.get("seed_dir")
    if (isinstance(seed, str) and seed.startswith("seed_") and seed[5:].isascii()
            and seed[5:].isdecimal() and len(seed[5:]) <= 10):
        seed = f"{int(seed[5:]):03d}"
    else:
        seed = "?"
    health = bundle.get("health", "unverified")
    if health not in ("ready", "collapsed", "unverified", "missing"):
        health = "unverified"
    return f"seed {seed} {health}"


class Arena:
    def __init__(self, args, targets):
        self.args, self.targets = args, targets
        self.cache = {}
        self.left_index, self.right_index = args.left - 1, args.right - 1
        self.left = self.load(self.left_index, args.left_seed)
        # A lesson never uses an opponent, so an unrelated failed checkpoint
        # must not block it. Resolve that selection only if the player leaves.
        self._deferred_right = bool(args.lesson)
        self.right = dict(self.left) if self._deferred_right else self.load(self.right_index, args.right_seed)
        self.mode = args.mode
        self.duel_enabled = args.duel
        self.duel_config = DuelConfig(args.round_ticks, args.warmup_ticks)
        self.lesson = None
        self.lesson_origin = None
        self.steps = 0
        self.reset()
        if args.lesson:
            self.start_lesson()

    def load(self, index, seed=None):
        # A bundle accepted under a research override must not bypass a later
        # healthy-only selection, even when a caller changes the policy.
        key = index, seed, bool(self.args.allow_unhealthy)
        if key not in self.cache:
            # NCA construction initializes CPU parameters before weights load.
            # A malformed checkpoint must not change an ongoing match's RNG
            # just because the player tried an unavailable culture.
            with torch.random.fork_rng(devices=[]):
                self.cache[key] = ensure_model(self.targets[index], self.args.device,
                    self.args.bootstrap_steps, preferred_seed=seed,
                    allow_unhealthy=self.args.allow_unhealthy)
        return self.cache[key]

    def reset(self):
        # Reset is a replay of this seed. Model loading must not perturb the match RNG.
        configure_runtime(seed=self.args.seed, cpu_threads=self.args.cpu_threads)
        self.size = active_grid_size(self.args.grid_size, self.left, self.right)
        if self.lesson:
            from lesson import Lesson
            self.a, self.b, self.owner, self.control = (
                torch.zeros_like(value) for value in reset_world(
                    self.size, self.args.device, self.left, self.right))
            center = self.size // 2
            self.start_positions = {"left": [center, center], "right": None}
            self.steps = 0
            self.duel = None
            self.lesson = Lesson(self.lesson.config, board_cells=self.size ** 2)
            return
        left_pos, right_pos = self.args.left_pos, self.args.right_pos
        # Equal border distance removes the sandbox's random placement skew.
        # Explicit positions remain available for clearly labeled custom rounds.
        if self.duel_enabled:
            center, start = (self.size - 1) // 2, self.size // 3
            left_pos = left_pos or (start, center)
            right_pos = right_pos or (self.size - 1 - start, center)
        self.a, self.b, self.owner, self.control = reset_world(
            self.size, self.args.device, self.left, self.right,
            left_pos=left_pos, right_pos=right_pos)
        self.steps = 0
        self.start_positions = {
            side: [int(pos) for pos in state[0, 3].nonzero()[0].flip(0).tolist()]
            for side, state in (("left", self.a), ("right", self.b))}
        self.duel = (DuelRound(self.duel_config, mode=self.mode, board_cells=self.size ** 2)
                     if self.duel_enabled else None)

    def set_duel(self, enabled):
        """Switch between an editable lab and a fresh scored round."""
        warning = self._leave_lesson()
        self.duel_enabled = bool(enabled)
        self.reset()
        return warning

    def start_lesson(self, config=None):
        """Start a fresh single-culture experiment with a calibrated 48px field."""
        from lesson import Lesson
        if self.lesson is None:
            self.lesson_origin = self.mode, self.args.grid_size
        self.lesson = Lesson(config, board_cells=48 ** 2)
        self.mode, self.args.grid_size = "soft", 48
        self.duel_enabled = False
        self.reset()

    def _leave_lesson(self):
        warning = None
        if self.lesson is not None:
            if self._deferred_right:
                try:
                    self.right = self.load(self.right_index, self.args.right_seed)
                except (ValueError, RuntimeError, OSError, IndexError):
                    name = self.left["name"].split("_", 1)[-1]
                    self.right, self.right_index = self.left, self.left_index
                    pin = self.args.right_seed
                    policy = "Auto selection kept." if pin is None else f"Pin {pin:03d} kept."
                    warning = f"Right unavailable; using {name} {checkpoint_caption(self.left)}. {policy}"
                self._deferred_right = False
            self.mode, self.args.grid_size = self.lesson_origin
            self.lesson, self.lesson_origin = None, None
        return warning

    def stop_lesson(self):
        """Return to a fresh sandbox with its previous rules and grid setting."""
        warning = self._leave_lesson()
        self.duel_enabled = False
        self.reset()
        return warning

    def plant_lesson(self):
        if self.lesson is None or self.lesson.phase != "seed":
            raise ValueError("Plant the lesson seed only at its seed stage")
        center = self.size // 2
        self.a = torch.zeros_like(self.a)
        self.a[0, 3:5, center, center] = 1
        self.owner = torch.zeros_like(self.owner)
        self.control = torch.zeros_like(self.control)
        self.owner[0, 0, center, center] = 1
        self.control[0, 0, center, center] = 1
        self.lesson.start_seed()

    def cut_lesson(self):
        if self.lesson is None or self.lesson.phase != "damage":
            raise ValueError("Apply the planned cut only at its damage stage")
        plan = self.lesson.snapshot()["plan"]
        if plan is None:
            raise ValueError("No safe lesson cut is available")
        values = crater(self.a, self.b, self.owner, self.control,
                        plan["x"], plan["y"], plan["radius"])
        remaining = int((values[0][:, 3:4] > .1).sum().item())
        try:
            self.lesson.apply_damage(remaining)
        except ValueError:
            self.lesson.invalidate("The applied cut did not match its preview; no valid observation.")
            return
        self.a, self.b, self.owner, self.control = values

    def watch_lesson(self):
        if self.lesson is None:
            raise ValueError("No lesson is active")
        self.lesson.watch()

    def _require_sandbox(self):
        if self.duel_enabled:
            raise ValueError("Duel edits are locked. Switch to LAB to plant, damage, or clear.")
        if self.lesson:
            raise ValueError("Use the lesson's marked seed or cut. Exit the lesson for free editing.")

    def select(self, index, side):
        if self.lesson and side != 0:
            raise ValueError("The solo lesson uses the left culture")
        bundle = self.load(index, self.args.left_seed if side == 0 else self.args.right_seed)
        if side == 0:
            self.left, self.left_index = bundle, index
        else:
            self.right, self.right_index = bundle, index
        self.reset()

    def clear(self):
        self._require_sandbox()
        self.a = torch.zeros_like(self.a)
        self.b = torch.zeros_like(self.b)
        self.owner = torch.zeros_like(self.owner)
        self.control = torch.zeros_like(self.control)
        self.steps = 0

    def damage(self, x, y, radius):
        self._require_sandbox()
        self.a, self.b, self.owner, self.control = crater(
            self.a, self.b, self.owner, self.control, x, y, radius)

    def plant(self, x, y, side):
        self._require_sandbox()
        x, y = clamp(x, 0, self.size - 1), clamp(y, 0, self.size - 1)
        # Clone because NCA inference returns inference tensors, which cannot be
        # modified outside inference_mode. Planting also clears the opposing spark.
        self.a, self.b = self.a.clone(), self.b.clone()
        self.owner, self.control = self.owner.clone(), self.control.clone()
        own, other = (self.a, self.b) if side == 0 else (self.b, self.a)
        if self.mode == "hard":
            other[:, :, y, x] = 0
        own[:, :, y, x] = 0
        own[0, 3:5, y, x] = 1
        self.owner[0, 0, y, x] = side + 1
        self.control[0, 0, y, x] = 1 if side == 0 else -1

    def step(self, count=1):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("step count must be a nonnegative integer")
        if self.lesson:
            return self._step_lesson(count)
        count = self.duel.clip_steps(count) if self.duel else count
        if self.duel and self.duel.mode != self.mode:
            raise ValueError("reset the round after changing its mode")
        advanced = 0
        with torch.inference_mode():
            for _ in range(count):
                if self.mode == "hard":
                    try:
                        self.a, self.b, self.owner, self.control = clash_step(
                            self.a, self.b, self.owner, self.control,
                            self.left["model"], self.right["model"],
                            pressure_gain=self.args.pressure_gain,
                            control_decay=self.args.control_decay,
                            capture_threshold=self.args.capture_threshold,
                            release_threshold=self.args.release_threshold,
                            tie_margin=self.args.tie_margin, strict_finite=bool(self.duel))
                    except FloatingPointError:
                        if not self.duel:
                            raise
                        # A rejected raw proposal leaves the last valid board
                        # intact, but the attempted tick invalidates its score.
                        self.steps += 1
                        advanced += 1
                        self.duel.record_tick(self.steps, None, None, finite=False)
                        break
                else:
                    self.a = self.left["model"](self.a)
                    self.b = self.right["model"](self.b)
                self.steps += 1
                advanced += 1
                if self.duel:
                    finite = bool(torch.isfinite(self.a).all() & torch.isfinite(self.b).all()
                                  & torch.isfinite(self.control).all())
                    if self.mode == "hard":
                        left, right = (int((self.owner == side).sum().item()) for side in (1, 2))
                    else:
                        left, right = (int((state[:, 3:4] > .1).sum().item()) for state in (self.a, self.b))
                    self.duel.record_tick(self.steps, left, right, finite=finite)
                    if self.duel.finished:
                        break
        return advanced

    def _step_lesson(self, count):
        """Advance only the visible culture, stopping at every teaching gate."""
        from lesson import choose_crater
        advanced = 0
        with torch.inference_mode():
            for _ in range(self.lesson.clip_steps(count)):
                # An empty opponent would still consume stochastic update RNG.
                # Solo inference matches the measured single-culture lesson.
                if not bool(torch.isfinite(self.a).all()):
                    self.steps += 1
                    advanced += 1
                    self.lesson.record_tick(self.steps, None, finite=False)
                    break
                proposed = self.left["model"](self.a)
                if proposed.shape != self.a.shape:
                    raise ValueError("the lesson model must preserve its state shape")
                self.steps += 1
                advanced += 1
                if not bool(torch.isfinite(proposed).all()):
                    self.lesson.record_tick(self.steps, None, finite=False)
                    break
                self.a = proposed
                living = int((self.a[:, 3:4] > .1).sum().item())
                self.lesson.record_tick(self.steps, living)
                if self.lesson.phase == "damage":
                    plan = choose_crater((self.a[0, 3] > .1).detach().cpu().numpy())
                    self.lesson.set_crater(plan)
                if not self.lesson.can_step:
                    break
        return advanced

    def stats(self):
        a, b = self.a[:, 3:4], self.b[:, 3:4]
        return {"step": self.steps, "mode": self.mode, "grid_size": self.size,
                "left_alive": int((a > .1).sum().item()),
                "right_alive": int((b > .1).sum().item()),
                "left_territory": int((self.owner == 1).sum().item()) if self.mode == "hard" else None,
                "right_territory": int((self.owner == 2).sum().item()) if self.mode == "hard" else None,
                "contested": int(((a > .1) & (b > .1)).sum().item()),
                "finite": bool(torch.isfinite(self.a).all() & torch.isfinite(self.b).all())}

    def rgb(self, team_colors=False, view="organisms"):
        if view == "pressure" and self.mode == "hard":
            c = self.control[0, 0].detach().cpu()
            a, b = c.clamp_min(0), (-c).clamp_min(0)
            rgb = torch.stack((a * .365 + b * .965, a * .886 + b * .518, a * .698 + b * .580))
        elif team_colors or view == "territory":
            a = self.a[0, 3].detach().cpu().clamp(0, 1)
            b = self.b[0, 3].detach().cpu().clamp(0, 1)
            if view == "territory" and self.mode == "hard":
                a = (self.owner[0, 0].detach().cpu() == 1).float()
                b = (self.owner[0, 0].detach().cpu() == 2).float()
            rgb = torch.stack((a * .365 + b * .965, a * .886 + b * .518, a * .698 + b * .580)).clamp(0, 1)
        elif self.mode == "hard":
            rgba = compose_rgba(self.a, self.b, self.owner)[0].detach().cpu().clamp(0, 1)
            rgb = rgba[:3] * rgba[3:4]
        else:
            # Symmetric additive premultiplied composition: no left-side visual bias.
            a, b = self.a[0, :4].detach().cpu().clamp(0, 1), self.b[0, :4].detach().cpu().clamp(0, 1)
            rgb = (a[:3] * a[3:4] + b[:3] * b[3:4]).clamp(0, 1)
        return (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def report(self, elapsed):
        return {**self.stats(), "seed": self.args.seed, "device": self.args.device,
                "cpu_threads": torch.get_num_threads(), "elapsed_seconds": elapsed,
                "steps_per_second": self.steps / elapsed if elapsed > 0 else 0,
                "left": {k: self.left.get(k) for k in ("name", "source", "seed_dir", "score", "health")},
                "right": None if self.lesson else {k: self.right.get(k) for k in ("name", "source", "seed_dir", "score", "health")},
                "duel": self.duel.snapshot() if self.duel else None,
                "lesson": self.lesson.snapshot() if self.lesson else None,
                "placement": "solo-center" if self.lesson else "custom" if self.args.left_pos or self.args.right_pos else
                             ("mirrored" if self.duel else "seeded-random"),
                "starting_positions": self.start_positions,
                "rules": {k: getattr(self.args, k) for k in ("pressure_gain", "control_decay",
                    "capture_threshold", "release_threshold", "tie_margin")}}

    def duel_recipe(self):
        """Capture the finished live round, never stale launch selections."""
        from duel_recipe import build_recipe

        states = (self.a, self.b, self.owner, self.control)
        if any(value.device.type != "cpu" for value in states):
            raise ValueError("Duel recipes currently support CPU rounds only.")
        report = self.report(0)
        report["finite"] = all(bool(torch.isfinite(value).all())
                               for value in states)
        runtime = runtime_info()
        runtime["device"] = self.args.device
        return build_recipe(report, runtime)


def parse_args(argv=None, default_mode="hard"):
    parser = argparse.ArgumentParser(description="Petri Clash: a living neural arena. No training required.")
    parser.add_argument("--mode", choices=("hard", "soft"), default=default_mode)
    parser.add_argument("--duel", action="store_true", help="play a fixed-length scored round with editing locked")
    parser.add_argument("--lesson", action="store_true", help="start the optional interactive solo regrowth lesson")
    parser.add_argument("--round-ticks", type=int, default=600, help="duel length in simulation ticks, including warmup")
    parser.add_argument("--warmup-ticks", type=int, default=60, help="initial duel growth ticks excluded from scoring")
    parser.add_argument("--grid-size", type=int, default=0, help="0 uses checkpoint grid size")
    parser.add_argument("--window-size", type=int, default=1100, help="window width; minimum 860")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--left", type=int, default=1)
    parser.add_argument("--right", type=int, default=2)
    parser.add_argument("--left-seed", type=int)
    parser.add_argument("--right-seed", type=int)
    parser.add_argument("--left-pos", type=parse_grid_pos)
    parser.add_argument("--right-pos", type=parse_grid_pos)
    parser.add_argument("--pressure-gain", type=float, default=.28)
    parser.add_argument("--control-decay", type=float, default=.97)
    parser.add_argument("--capture-threshold", type=float, default=.55)
    parser.add_argument("--release-threshold", type=float, default=.12)
    parser.add_argument("--tie-margin", type=float, default=.01)
    parser.add_argument("--team-colors", action="store_true")
    parser.add_argument("--reduced-motion", action="store_true", help="disable visual easing and moving effects")
    parser.add_argument("--crater-radius", type=int, default=4)
    parser.add_argument("--steps-per-frame", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--bootstrap-steps", type=int, default=0, help="explicitly opt into training missing weights")
    parser.add_argument("--allow-unhealthy", action="store_true", help="allow collapsed/unverified checkpoints for research")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, default=0, help="replayable simulation RNG seed")
    parser.add_argument("--headless-frames", type=int, default=0, help="simulate up to N steps (duels stop at their endpoint), with no SDL or FPS sleep")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--report", type=Path, help="write final machine-readable match report")
    parser.add_argument("--snapshot", type=Path, help="save final PNG (board headlessly; entire interactive window)")
    parser.add_argument("--export-duel", type=Path, help="save a portable recipe for the final completed CPU duel")
    parser.add_argument("--duel-save-dir", type=Path, default=Path("duels"),
                        help="folder for SAVE DUEL / S recipes (default: ./duels)")
    parser.add_argument("--ui-frames", type=int, default=0, help="exit interactive mode after N frames, useful for UI smoke tests")
    args = parser.parse_args(argv)
    outputs = [(name, getattr(args, name)) for name in ("report", "snapshot", "export_duel")
               if getattr(args, name) is not None]
    if len(outputs) > 1:
        from duel_recipe import paths_alias
        try:
            for i, (name, path) in enumerate(outputs):
                for other_name, other_path in outputs[i + 1:]:
                    if paths_alias(path, other_path):
                        parser.error(f"--{name.replace('_', '-')} and --{other_name.replace('_', '-')} need different output files")
        except (OSError, RuntimeError):
            parser.error("could not resolve output locations; choose accessible, distinct files")
    if args.lesson and (args.duel or args.headless_frames):
        parser.error("--lesson is interactive and cannot be combined with --duel or --headless-frames")
    try:
        DuelConfig(args.round_ticks, args.warmup_ticks)
    except ValueError as exc:
        parser.error(str(exc))
    if args.grid_size and args.grid_size < 8:
        parser.error("--grid-size must be 0 or at least 8")
    if args.fps < 1 or args.crater_radius < 1 or args.window_size < 1:
        parser.error("--fps, --crater-radius and --window-size must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if min(args.headless_frames, args.ui_frames, args.bootstrap_steps) < 0:
        parser.error("frame counts, bootstrap steps and seed must be non-negative")
    if any(x is not None and x < 0 for x in (args.left_seed, args.right_seed)):
        parser.error("checkpoint seeds must be non-negative")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    values = (args.pressure_gain, args.control_decay, args.capture_threshold,
              args.release_threshold, args.tie_margin)
    if not all(math.isfinite(x) for x in values):
        parser.error("battle parameters must be finite")
    if not (0 <= args.release_threshold < args.capture_threshold <= 1):
        parser.error("require 0 <= release threshold < capture threshold <= 1")
    if args.pressure_gain <= 0 or not 0 <= args.control_decay <= 1 or not 0 <= args.tie_margin <= 1:
        parser.error("pressure gain must be positive, control decay in [0,1], tie margin in [0,1]")
    return args


BG, PANEL, BORDER = (23, 27, 29), (35, 42, 45), (65, 77, 80)
TEXT, MUTED = (232, 228, 216), (169, 177, 175)
GREEN, CORAL, AMBER = (93, 226, 178), (246, 132, 148), (217, 182, 111)


class ArenaUI:
    """Presentation-only animation around an exact, seeded simulation."""
    def __init__(self, arena):
        import pygame
        from feedback import FrameBlend, ScoreTracker
        self.pg, self.arena = pygame, arena
        pygame.display.init()
        pygame.font.init()
        width = max(860, arena.args.window_size)
        self.window = pygame.display.set_mode((width, max(740, int(width * .82))), pygame.RESIZABLE)
        pygame.display.set_caption("Petri Clash | Living arena")
        self.fonts = {s: pygame.font.SysFont(
            "DejaVu Sans Mono,Consolas,monospace" if s in (12, 36) else
            "DejaVu Sans Condensed,Arial,sans" if s in (18, 22, 28) else "DejaVu Sans,Arial,sans",
            s, bold=s in (18, 22, 28, 36)) for s in (11, 12, 13, 14, 16, 18, 22, 28, 36)}
        self.button_font = pygame.font.SysFont("DejaVu Sans Condensed,Arial,sans", 13, bold=True)
        self.status = []
        self._picker_status_cache = {}
        self._picker_status_key = None
        # Picker tiles stay 28px at every window size; prepare them once.
        self.thumbs = [pygame.transform.smoothscale(
            pygame.image.load(str(t)).convert_alpha(), (28, 28)) for t in arena.targets]
        self.paused, self.side, self.view = False, 0, "organisms"
        self.pointer_tool = "damage"
        self.speed, self.radius = arena.args.steps_per_frame, arena.args.crater_radius
        self.team_colors = arena.args.team_colors
        self.reduced_motion = arena.args.reduced_motion
        self.message = "Duel: sustain more cells over time. Pausing freezes the clock." if arena.duel else "Grow. Collide. Regenerate."
        self.buttons = []
        self.result_buttons = []
        self.result_rect = None
        self.stale_result_rect = None
        self.result_visible = True
        self.saved_duel = None
        self.saved_duel_path = None
        self.board = pygame.Rect(0, 0, 1, 1)
        self._scaled_board = None
        self._scaled_board_rgb = None
        self._scaled_board_size = None
        self.clock = pygame.time.Clock()
        self.stats_cache = arena.stats()
        self.current_fps = 0.0
        self.blend = FrameBlend()
        self.score = ScoreTracker()
        self.effects = []
        self.notices = []
        self.last_draw = time.perf_counter()
        self.elapsed = 0.0
        self.duel_snapshot = None
        self.lesson_snapshot = None
        self.lesson_credit = 0.0
        self.lesson_gate_pending = False
        self.lesson_notice_phase = None
        self.held_shift_keys = set()
        self.lesson_prior_speed = self.speed if arena.lesson else None
        if arena.lesson:
            self.speed = 1
        self.refresh(reset=True)
        if not arena.lesson and not arena.duel and self.radius != arena.args.crater_radius:
            self.notice(f"Damage radius limited to {self.radius} cells for this field.", AMBER)

    def picker_policy(self):
        side = 0 if self.arena.lesson else self.side
        pin = self.arena.args.left_seed if side == 0 else self.arena.args.right_seed
        return pin, bool(self.arena.args.allow_unhealthy)

    def refresh_picker_status(self, *, force=False):
        """Read selection-aware metadata on policy changes, never every frame."""
        key = self.picker_policy()
        if force:
            self._picker_status_cache.clear()
            self._picker_status_key = None
        if key != self._picker_status_key:
            if key not in self._picker_status_cache:
                pin, allow_unhealthy = key
                self._picker_status_cache[key] = [
                    target_status(target, preferred_seed=pin, allow_unhealthy=allow_unhealthy)
                    for target in self.arena.targets]
            self.status = self._picker_status_cache[key]
            self._picker_status_key = key

    def loaded_checkpoint_summary(self):
        left = f"L {checkpoint_caption(self.arena.left)}"
        if self.arena.lesson:
            return left
        return f"{left} / R {checkpoint_caption(self.arena.right)}"

    def text(self, text, x, y, size=14, color=TEXT, width=None):
        text = str(text)
        if width is not None:
            while text and self.fonts[size].size(text)[0] > width:
                text = text[:-2] + "…" if len(text) > 1 else ""
        self.window.blit(self.fonts[size].render(text, True, color), (x, y))

    def button(self, rect, label, action, active=False, color=AMBER):
        pg = self.pg
        rect = pg.Rect(rect)
        hover = rect.collidepoint(pg.mouse.get_pos())
        fill = (63, 66, 59) if active else ((62, 72, 75) if hover else (52, 62, 66))
        pg.draw.rect(self.window, fill, rect, border_radius=2)
        pg.draw.line(self.window, color if active else (85, 95, 96), rect.topleft, (rect.right - 1, rect.top))
        pg.draw.line(self.window, (12, 16, 18), rect.bottomleft, rect.bottomright)
        if active:
            pg.draw.rect(self.window, color, (rect.x, rect.y, 3, rect.height))
        surface = self.button_font.render(label, True, color if active else TEXT)
        self.window.blit(surface, surface.get_rect(center=rect.center))
        self.buttons.append((rect, action))

    def refresh(self, reset=False):
        self.refresh_picker_status()
        if not self.arena.lesson and not self.arena.duel:
            self.radius = clamp(self.radius, 1, self.arena.size)
        self.stats_cache = self.arena.stats()
        if reset:
            if self.result_rect and self.result_visible and self.duel_snapshot and self.duel_snapshot["finished"]:
                self.stale_result_rect = self.result_rect.copy()
            from feedback import FrameBlend
            self.blend = FrameBlend()
            self.score.reset()
            self.effects.clear()
            self.notices.clear()
            self.result_visible = True
            self.lesson_credit = 0.0
            self.lesson_gate_pending = bool(self.arena.lesson)
            self.lesson_notice_phase = None
        self.summary = self.score.update(self.stats_cache)
        self.duel_snapshot = self.arena.duel.snapshot() if self.arena.duel else None
        before_phase = self.lesson_snapshot["phase"] if self.lesson_snapshot else None
        self.lesson_snapshot = self.arena.lesson.snapshot() if self.arena.lesson else None
        if self.lesson_snapshot:
            if self.lesson_snapshot["phase"] != before_phase:
                self.lesson_credit = 0.0
                if self.lesson_snapshot["phase"] in ("damage", "complete", "timeout", "unavailable", "invalid"):
                    # Teaching checkpoints should show their exact frozen board,
                    # with no old seed/cut markers or blend from an earlier phase.
                    self.effects.clear()
                    self.blend.reset()
            if not self.lesson_snapshot["can_step"]:
                self.paused = True
        if self.duel_snapshot and self.duel_snapshot["finished"]:
            self.paused = True

    def notice(self, text, color=TEXT):
        self.message = text
        self.lesson_notice_phase = self.arena.lesson.phase if self.arena.lesson else None
        self.notices.append((text, color, self.elapsed))
        self.notices = self.notices[-3:]

    def save_duel(self):
        """Save once per finished round without advancing or changing it."""
        from duel_recipe import save_unique_recipe

        try:
            if (self.saved_duel is self.arena.duel and self.saved_duel_path is not None
                    and self.saved_duel_path.is_file()):
                self.notice(f"Already saved: {self.saved_duel_path.name}", AMBER)
                return
            recipe = self.arena.duel_recipe()
            path = save_unique_recipe(self.arena.args.duel_save_dir, recipe)
        except (ValueError, OSError, RuntimeError) as exc:
            self.notice(f"Could not save duel: {exc}", CORAL)
            return
        self.saved_duel, self.saved_duel_path = self.arena.duel, path
        self.notice(f"Saved duel: {path.name}", AMBER)
        print(f"Saved duel recipe: {path}", file=sys.stderr)

    def effective_pointer_tool(self, button=1):
        """Resolve temporary shortcuts without changing the persistent lab tool."""
        if button == 3:
            return "plant_right"
        if button != 1:
            return None
        if self.held_shift_keys:
            return "plant_left"
        # Guided/scored play retains its original click contract, even when
        # lab rectangles or a previously selected planting tool still exist.
        if self.arena.lesson or self.arena.duel:
            return "damage"
        return self.pointer_tool

    def pointer_cell(self, pos):
        """Use the same discrete field location for preview and dispatch."""
        return (clamp(int((pos[0] - self.board.x) * self.arena.size / self.board.width), 0, self.arena.size - 1),
                clamp(int((pos[1] - self.board.y) * self.arena.size / self.board.height), 0, self.arena.size - 1))

    def change_radius(self, delta):
        self.radius = clamp(self.radius + delta, 1, self.arena.size)

    def interact(self, x, y, side=None):
        """Apply one input, then report its real immediate effect, even paused."""
        if self.arena.lesson:
            marker = self.lesson_marker()
            phase = self.arena.lesson.phase
            if marker:
                gx, gy, radius = marker
                hit_radius = 2 if phase == "seed" else radius
                inside = (x - gx) ** 2 + (y - gy) ** 2 <= hit_radius ** 2
                if inside and ((phase == "seed" and side == 0) or (phase == "damage" and side is None)):
                    self.lesson_primary()
                    return
            self.notice("Use the marked seed or cut, or press Enter. G returns to free editing.", AMBER)
            return
        if self.arena.duel:
            self.notice("Duel edits are locked. BACK TO LAB restores planting and damage.", AMBER)
            return
        from feedback import FrameBlend
        before = self.arena.stats()
        if side is None:
            self.arena.damage(x, y, self.radius)
            after = self.arena.stats()
            left = before["left_alive"] - after["left_alive"]
            right = before["right_alive"] - after["right_alive"]
            label = f"DAMAGE  L -{left} / R -{right} life"
            color, radius, kind = (255, 209, 139), self.radius, "damage"
            detail = f"Crater ({x}, {y}): left -{left}, right -{right} living cells."
        else:
            self.arena.plant(x, y, side)
            label = "LEFT SEED +" if side == 0 else "RIGHT SEED +"
            color, radius, kind = (GREEN if side == 0 else CORAL), 1, "plant"
            detail = f"{'Left' if side == 0 else 'Right'} seed planted at ({x}, {y}). Growth needs simulation ticks."
        self.effects.append({"x": x, "y": y, "radius": radius, "kind": kind,
                             "color": color, "label": label, "born": self.elapsed})
        self.effects = self.effects[-24:]
        self.blend = FrameBlend()  # Edits are visible immediately, never ghosted over.
        self.refresh()
        self.notice(detail, color)

    def draw_score(self, width):
        if self.arena.lesson:
            self.draw_lesson_score()
            return
        pg, st = self.pg, self.stats_cache
        sx, y = self.rail_x, 111
        self.score_rect = pg.Rect(sx, y, 286, 181)
        hard = self.arena.mode == "hard"
        left, right = self.summary["left"], self.summary["right"]
        duel = self.duel_snapshot
        metric = "HELD CELLS" if hard else "LIVING CELLS"
        self.text(("MEAN " if duel else "") + metric, sx + 12, y, 12, AMBER)
        state = "FINAL" if duel and duel["finished"] else "PAUSED" if self.paused else "LIVE"
        self.text(state, sx + 222, y, 11, MUTED)
        for side, value, color in ((0, left, GREEN), (1, right, CORAL)):
            x = sx + 12 + side * 140
            name = (self.arena.left if side == 0 else self.arena.right)["name"].split("_", 1)[-1]
            self.text(f"{'L' if side == 0 else 'R'} / {name.upper()}", x, y + 24, 12, color, 126)
            average = duel["averages"]["left" if side == 0 else "right"] if duel else None
            number = (f"{average:,.1f}" if average is not None else "—") if duel else f"{value:,}"
            size = 36 if self.fonts[36].size(number)[0] <= 126 else 28
            self.text(number, x - 2, y + 40, size, color, 128)
            alive = st["left_alive" if side == 0 else "right_alive"]
            pct = self.summary["percent"]["left" if side == 0 else "right"]
            self.text(f"{value:,} now" if duel else f"{pct:.1f}% held" if hard else f"{pct:.1f}% alive", x, y + 83, 13, MUTED)
            self.text(f"{duel['scored_ticks']} scored ticks" if duel else f"{alive} living cells", x, y + 100, 13, MUTED, 128)
        if duel:
            headline = self.duel_headline()
            detail = ("Growth warmup / no points yet" if duel["phase"] == "warmup" else
                      "Equal cell-ticks = draw" if duel["winner"] == "draw" else
                      "Winner: most cell-ticks held" if hard else "Winner: most living cell-ticks")
        else:
            headline = self.summary["headline"]
            detail = self.summary["trend"] if self.summary["span"] else "Open-ended / no final winner"
        self.text(headline, sx + 12, y + 124, 16, TEXT, 262)
        self.text(detail, sx + 12, y + 146, 11, MUTED, 262)
        total, bar_width = self.arena.size ** 2, 262
        self.score_bars = []
        if hard:
            bar = pg.Rect(sx + 12, y + 170, bar_width, 7)
            self.score_bars.append(bar)
            pg.draw.rect(self.window, (15, 19, 21), bar)
            lw, rw = round(bar_width * left / total), round(bar_width * right / total)
            if lw:
                pg.draw.rect(self.window, GREEN, (bar.x, bar.y, lw, bar.height))
            if rw:
                pg.draw.rect(self.window, CORAL, (bar.right - rw, bar.y, rw, bar.height))
        else:
            for i, value, color in ((0, left, GREEN), (1, right, CORAL)):
                bar = pg.Rect(sx + 12, y + 167 + i * 9, bar_width, 4)
                self.score_bars.append(bar)
                pg.draw.rect(self.window, (15, 19, 21), bar)
                filled = round(bar_width * value / total)
                if filled:
                    pg.draw.rect(self.window, color, (bar.x, bar.y, filled, 4))

    def lesson_instruction(self):
        phase = self.arena.lesson.phase
        return {
            "seed": ("Plant at the marked crosshair.", "Shift-click it or press Enter."),
            "grow": (f"Grow to {self.arena.lesson.config.grow_ticks} simulation ticks.", "Space pauses; N advances one tick."),
            "damage": ("Inspect the highlighted practice cut.", "Click the cut or Enter to apply it."),
            "injured": ("The cut is frozen for inspection.", "Space / Enter watches; N steps."),
            "recover": (f"Slow watch: {6 * self.speed} ticks/s.", f"Hold the count goal for {self.arena.lesson.config.hold_ticks} ticks."),
            "complete": ("Living-cell recovery observed.", "Press R to repeat the experiment."),
            "timeout": ("The count target was not held.", "R retries; choose another culture."),
            "unavailable": ("No safe practice cut for this state.", "R retries; choose another culture."),
            "invalid": ("This observation is not valid.", "Try a ready culture or restart."),
        }[phase]

    def lesson_marker(self):
        if not self.arena.lesson:
            return None
        if self.arena.lesson.phase == "seed":
            return self.arena.size // 2, self.arena.size // 2, 1
        if self.arena.lesson.phase == "damage":
            plan = self.arena.lesson.snapshot()["plan"]
            if plan:
                return plan["x"], plan["y"], plan["radius"]
        return None

    def lesson_primary(self):
        if not self.arena.lesson or self.lesson_gate_pending:
            return
        from feedback import FrameBlend
        phase = self.arena.lesson.phase
        marker = self.lesson_marker()
        if phase == "seed":
            self.arena.plant_lesson()
            self.paused = False
            self.notice("Seed planted. Watch one culture grow; Space pauses and N steps.", GREEN)
            label, kind, color = "LESSON SEED", "plant", GREEN
        elif phase == "damage":
            plan = self.arena.lesson.snapshot()["plan"]
            self.arena.cut_lesson()
            self.paused = True
            if not self.arena.lesson.snapshot()["valid"]:
                self.refresh()
                return
            self.notice(f"Cut removed {plan['removed']} living cells. Space watches regrowth; N steps.", AMBER)
            label, kind, color = f"CUT -{plan['removed']} LIFE", "damage", AMBER
        elif phase == "injured":
            self.arena.watch_lesson()
            self.paused = False
            self.lesson_credit = 0.0
            self.lesson_gate_pending = True
            self.notice(f"Slow observation: {6 * self.speed} simulation ticks per second. N inspects one tick.")
            self.refresh()
            return
        elif self.arena.lesson.finished:
            self.action("reset")
            return
        else:
            return
        self.lesson_gate_pending = True
        self.blend = FrameBlend()
        if marker:
            x, y, radius = marker
            self.effects.append({"x": x, "y": y, "radius": radius, "kind": kind,
                                 "color": color, "label": label, "born": self.elapsed})
        self.refresh()

    def step_budget(self, elapsed_seconds):
        """Slow lesson observation without slowing rendering or scoring by wall time."""
        if self.paused:
            self.lesson_credit = 0.0
            return 0
        if self.arena.lesson and self.arena.lesson.phase == "recover":
            elapsed = float(elapsed_seconds)
            elapsed = min(.25, max(0.0, elapsed)) if math.isfinite(elapsed) else 0.0
            self.lesson_credit += elapsed * 6 * self.speed
            count = int(self.lesson_credit)
            self.lesson_credit -= count
            return count
        self.lesson_credit = 0.0
        return self.speed

    def draw_lesson_score(self):
        pg, lesson = self.pg, self.lesson_snapshot
        sx, y = self.rail_x, 111
        self.score_rect = pg.Rect(sx, y, 286, 181)
        self.score_bars = []
        phase = lesson["phase"]
        name = self.arena.left["name"].split("_", 1)[-1].upper()
        self.text("SOLO / " + name, sx + 12, y, 12, AMBER, 206)
        self.text("PAUSED" if self.paused else "LIVE", sx + 222, y, 11, MUTED)
        stages = {"seed": "01 / PLANT", "grow": "02 / GROW", "damage": "03 / PREVIEW CUT",
                  "injured": "04 / INSPECT CUT", "recover": "05 / OBSERVE", "complete": "RECOVERY OBSERVED",
                  "timeout": "OBSERVATION ENDED", "unavailable": "NO SAFE CUT", "invalid": "INVALID OBSERVATION"}
        color = GREEN if phase == "complete" else CORAL if phase == "invalid" else TEXT
        self.text(stages[phase], sx + 12, y + 23, 18, color, 262)
        current = lesson["current"]
        value = "0" if phase == "seed" else "—" if current is None else f"{current:,}"
        self.text(value, sx + 10, y + 47, 36, GREEN, 128)
        self.text("living cells", sx + 12, y + 91, 12, MUTED)
        percent = 100 * lesson["config"]["target_num"] / lesson["config"]["target_den"]
        goal_label = f"{percent:.0f}% goal" if percent.is_integer() else f"{percent:.1f}% goal"
        for row, (label, value) in enumerate((("Baseline", lesson["baseline"]), (goal_label, lesson["target"]))):
            self.text(label, sx + 156, y + 48 + row * 27, 11, MUTED)
            self.text("—" if value is None else str(value), sx + 223, y + 46 + row * 27, 16, TEXT, 50)
        if phase in ("seed", "grow"):
            progress = f"Growth {lesson['growth_ticks']} / {lesson['config']['grow_ticks']} ticks"
            fraction = lesson["growth_ticks"] / lesson["config"]["grow_ticks"]
        elif phase == "damage":
            plan = lesson["plan"]
            progress = f"Cut removes {plan['removed']} / {plan['baseline']} cells" if plan else "Preparing a safe practice cut"
            fraction = 0.0
        else:
            progress = f"Recovery {lesson['recovery_ticks']} ticks / hold {lesson['hold']} of {lesson['required_hold']}"
            fraction = min(1.0, (current or 0) / lesson["baseline"]) if lesson["baseline"] else 0.0
        self.text(progress, sx + 12, y + 114, 12, MUTED, 262)
        button = {"seed": "PLANT SEED / ENTER", "damage": "APPLY CUT / ENTER",
                  "injured": "WATCH REGROWTH / SPACE"}.get(phase)
        if lesson["finished"]:
            button = "RETRY LESSON / R"
        if button:
            self.button((sx + 12, y + 139, 262, 27), button, "lesson_primary", True)
        else:
            self.text("Space: pause / N: inspect one tick", sx + 12, y + 144, 12, MUTED, 262)
        bar = pg.Rect(sx + 12, y + 175, 262, 3)
        self.score_bars.append(bar)
        pg.draw.rect(self.window, BORDER, bar)
        if fraction > 0:
            pg.draw.rect(self.window, GREEN, (bar.x, bar.y, round(bar.width * fraction), bar.height))

    def draw_lesson_cue(self):
        marker = self.lesson_marker()
        if marker is None:
            return
        pg, size = self.pg, self.arena.size
        x, y, radius = marker
        scale = self.board.width / size
        center = (self.board.x + (x + .5) * scale, self.board.y + (y + .5) * scale)
        color = GREEN if self.arena.lesson.phase == "seed" else AMBER
        if self.arena.lesson.phase == "damage":
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            rgba = np.zeros((size, size, 4), dtype=np.uint8)
            rgba[mask] = (*AMBER, 75)
            overlay = pg.image.frombuffer(rgba.tobytes(), (size, size), "RGBA")
            self.window.blit(pg.transform.scale(overlay, self.board.size), self.board)
        pg.draw.circle(self.window, color, center, max(10, round(radius * scale)), 2)
        pg.draw.line(self.window, color, (center[0] - 7, center[1]), (center[0] + 7, center[1]), 2)
        pg.draw.line(self.window, color, (center[0], center[1] - 7), (center[0], center[1] + 7), 2)

    def duel_headline(self):
        duel = self.duel_snapshot
        if duel["phase"] == "invalid":
            return "ROUND INVALID"
        if duel["finished"]:
            return "ROUND DRAW" if duel["winner"] == "draw" else f"{duel['winner'].upper()} WINS ROUND"
        if duel["phase"] == "warmup":
            return f"WARMUP / {duel['warmup_remaining_ticks']} LEFT"
        return f"{duel['remaining_ticks']} TICKS REMAIN"

    def draw_duel(self):
        """A small progress track, then a result that never hides its metric."""
        duel, pg = self.duel_snapshot, self.pg
        self.result_buttons = []
        if not duel:
            self.result_rect = None
            return
        track = pg.Rect(self.board.x + 12, self.board.bottom - 10, self.board.width - 24, 3)
        pg.draw.rect(self.window, BORDER, track)
        progress = round(track.width * duel["tick"] / duel["total_ticks"])
        if progress:
            pg.draw.rect(self.window, AMBER, (track.x, track.y, progress, track.height))
        if not duel["finished"] or not self.result_visible:
            self.result_rect = None
            return
        width = min(440, self.board.width - 32)
        self.result_rect = pg.Rect(0, 0, width, 258)
        self.result_rect.center = self.board.center
        rect = self.result_rect
        shadow = pg.Surface((rect.width + 12, rect.height + 12), pg.SRCALPHA)
        shadow.fill((0, 0, 0, 100))
        self.window.blit(shadow, (rect.x - 6, rect.y - 6))
        pg.draw.rect(self.window, PANEL, rect)
        pg.draw.rect(self.window, BORDER, rect, 1)
        pg.draw.line(self.window, AMBER, rect.topleft, rect.topright, 3)
        x, y = rect.x + 22, rect.y + 18
        self.text("SEEDED DUEL / FINAL", x, y, 12, AMBER)
        color = GREEN if duel["winner"] == "left" else CORAL if duel["winner"] == "right" else TEXT
        self.text(self.duel_headline(), x, y + 27, 28, color, width - 44)
        if duel["valid"]:
            self.text("AVERAGE " + duel["metric"].upper(), x, y + 73, 12, MUTED)
            self.text(f"L {duel['averages']['left']:,.2f}   /   R {duel['averages']['right']:,.2f}",
                      x, y + 96, 22, TEXT, width - 44)
            self.text(f"Cell-ticks: L {duel['scores']['left']:,} / R {duel['scores']['right']:,}",
                      x, y + 130, 12, MUTED, width - 44)
            self.text(f"Ticks {duel['warmup_ticks'] + 1}–{duel['total_ticks']} scored. Most wins; equal is a draw.",
                      x, y + 151, 11, MUTED, width - 44)
        else:
            self.text("Non-finite state. No winner awarded.", x, y + 87, 14, CORAL, width - 44)
            self.text("Try a healthy culture or another seed.", x, y + 112, 13, MUTED, width - 44)
        button_width = (width - 60) // 3
        start = len(self.buttons)
        self.button((rect.right - 72, rect.y + 13, 56, 23), "HIDE", "result")
        if duel["valid"]:
            saved = self.saved_duel is self.arena.duel and self.saved_duel_path is not None
            self.button((rect.right - 184, rect.y + 13, 104, 23),
                        "SAVED / S" if saved else "SAVE DUEL / S", "save_duel", saved)
        for i, (label, action) in enumerate((("REMATCH / R", "reset"), ("NEXT SEED", "next_seed"), ("BACK TO LAB", "duel"))):
            self.button((x + i * (button_width + 8), rect.bottom - 43, button_width, 28), label, action,
                        active=i == 0)
        self.result_buttons = self.buttons[start:]
        del self.buttons[start:]

    def draw_effects(self):
        pg, arena = self.pg, self.arena
        self.effects = [effect for effect in self.effects if self.elapsed - effect["born"] < 1.25]
        if not self.effects:
            return
        scale = self.board.width / arena.size
        overlay = pg.Surface(self.board.size, pg.SRCALPHA)
        for effect in self.effects:
            age = self.elapsed - effect["born"]
            progress = min(1.0, age / 1.25)
            opacity = round(230 * (1 - progress))
            center = ((effect["x"] + .5) * scale, (effect["y"] + .5) * scale)
            base = max(5, effect["radius"] * scale)
            growth = 0 if self.reduced_motion else progress * 16
            color = (*effect["color"], opacity)
            pg.draw.circle(overlay, color, center, round(base + growth), 2)
            if effect["kind"] == "damage":
                if not self.reduced_motion:
                    for i in range(8):
                        angle = i * math.tau / 8
                        a = (center[0] + math.cos(angle) * (base + growth + 3),
                             center[1] + math.sin(angle) * (base + growth + 3))
                        b = (a[0] + math.cos(angle) * 6, a[1] + math.sin(angle) * 6)
                        pg.draw.line(overlay, color, a, b, 2)
            else:
                pg.draw.line(overlay, color, (center[0] - 5, center[1]), (center[0] + 5, center[1]), 2)
                pg.draw.line(overlay, color, (center[0], center[1] - 5), (center[0], center[1] + 5), 2)
            label = self.fonts[12].render(effect["label"], True, effect["color"])
            label.set_alpha(opacity)
            tx = clamp(round(center[0] - label.get_width() / 2), 4, max(4, self.board.width - label.get_width() - 4))
            ty = clamp(round(center[1] - base - 26 - growth), 4, self.board.height - 24)
            pg.draw.rect(overlay, (12, 19, 23, round(220 * (1 - progress))), (tx - 3, ty - 2, label.get_width() + 6, 21), border_radius=4)
            overlay.blit(label, (tx, ty))
        self.window.blit(overlay, self.board)

    def draw_pointer_preview(self):
        pg, arena = self.pg, self.arena
        mouse = pg.mouse.get_pos()
        if arena.duel or arena.lesson or not self.board.collidepoint(mouse):
            return
        tool = self.effective_pointer_tool(3 if pg.mouse.get_pressed()[2] else 1)
        x, y = self.pointer_cell(mouse)
        scale = self.board.width / arena.size
        center = (round(self.board.x + (x + .5) * scale),
                  round(self.board.y + (y + .5) * scale))
        # Large craters and edge markers must not paint over the instrument rail.
        previous_clip = self.window.get_clip()
        self.window.set_clip(self.board)
        try:
            if tool == "damage":
                pg.draw.circle(self.window, AMBER, center, max(3, round(self.radius * scale)), 1)
                pg.draw.circle(self.window, AMBER, center, 2)
            else:
                color, label = (GREEN, "L") if tool == "plant_left" else (CORAL, "R")
                radius = max(5, round(scale * .4))
                pg.draw.circle(self.window, color, center, radius, 1)
                pg.draw.line(self.window, color, (center[0] - radius - 3, center[1]),
                             (center[0] + radius + 3, center[1]), 1)
                pg.draw.line(self.window, color, (center[0], center[1] - radius - 3),
                             (center[0], center[1] + radius + 3), 1)
                marker = self.fonts[12].render(label, True, color)
                label_x = clamp(center[0] + radius + 5, self.board.left + 2,
                                self.board.right - marker.get_width() - 2)
                label_y = clamp(center[1] - radius - marker.get_height() - 2,
                                # Leave the inset status/size captions readable near the top edge.
                                self.board.top + 11 + self.fonts[11].get_height() + 4,
                                self.board.bottom - marker.get_height() - 2)
                backdrop = pg.Rect(label_x - 2, label_y, marker.get_width() + 4, marker.get_height())
                pg.draw.rect(self.window, BG, backdrop)
                self.window.blit(marker, (label_x, label_y))
        finally:
            self.window.set_clip(previous_clip)

    def board_surface(self, array):
        """Reuse only identical paused rasters, leaving live drawing unchanged."""
        pg = self.pg
        if not self.paused:
            self._scaled_board = self._scaled_board_rgb = self._scaled_board_size = None
            surface = pg.surfarray.make_surface(array.swapaxes(0, 1))
            return pg.transform.scale(surface, self.board.size)
        size = self.board.size
        # Compare the actual post-blend pixels: paused edits and easing still draw.
        if self._scaled_board_size != size or not np.array_equal(array, self._scaled_board_rgb):
            surface = pg.surfarray.make_surface(array.swapaxes(0, 1))
            self._scaled_board = pg.transform.scale(surface, size)
            self._scaled_board_rgb = array.copy()
            self._scaled_board_size = size
        return self._scaled_board

    def draw(self, dt=None):
        pg, arena = self.pg, self.arena
        now = time.perf_counter()
        dt = min(.25, max(0.0, now - self.last_draw if dt is None else dt))
        self.last_draw = now
        self.elapsed += dt
        self.refresh()
        w, h = self.window.get_size()
        self.window.fill(BG)
        self.buttons = []
        self.stale_result_rect = None
        sidebar, by, gap = 286, 110, 16
        size = max(1, min(w - sidebar - 56, h - by - 52))
        bx = max(20, (w - size - gap - sidebar) // 2)
        sx = bx + size + gap
        self.rail_x = sx
        self.board = pg.Rect(bx, by, size, size)
        self.text("PETRI CLASH", bx, 13, 28)
        subtitle = "REGROWTH LESSON / SINGLE CULTURE" if arena.lesson else \
                   "SEEDED DUEL / SUSTAIN MORE LIFE" if arena.duel and arena.mode == "soft" else \
                   "SEEDED DUEL / HOLD MORE LAND" if arena.duel else "NEURAL GROWTH SANDBOX"
        self.text(subtitle, bx + 1, 49, 11, MUTED)
        self.button((self.board.right - 110, 22, 110, 27),
                    "BACK TO LAB" if arena.duel else "START DUEL", "duel", bool(arena.duel))
        self.button((self.board.right - 220, 22, 104, 27),
                    "EXIT LESSON" if arena.lesson else "LESSON / G", "lesson", bool(arena.lesson))
        self.text(f"SIM SEED {arena.args.seed:03d}", sx + 12, 23, 12, AMBER)
        self.text(f"TICK {arena.steps:05d} / {self.current_fps:.0f} FPS", sx + 12, 47, 11, MUTED)
        x = bx
        finished = bool(arena.duel and arena.duel.finished)
        pause_control = ("RESULT", "result", self.result_visible) if finished else \
                        ("RESUME" if self.paused else "PAUSE", "pause", self.paused)
        if arena.lesson:
            label = {"seed": "PLANT", "damage": "CUT", "injured": "WATCH"}.get(arena.lesson.phase)
            if arena.lesson.finished:
                pause_control = "RETRY", "reset", True
            elif label:
                pause_control = label, "lesson_primary", True
        for width, label, action, active in (
            (74, *pause_control),
            (68, "REPLAY", "reset", False), (48, "STEP", "step", False),
            (42, f"{self.speed}x", "speed", False),
            (66, "SOLO" if arena.lesson else "HARD" if arena.mode == "hard" else "SOFT", "mode", True),
            (72, "COLORS", "colors", self.team_colors),
            (60, "FX OFF" if self.reduced_motion else "FX ON", "motion", not self.reduced_motion)):
            self.button((x, 73, width, 27), label, action, active)
            x += width + 6
        # The board is the dominant element; the rail always sits 16px beside it.
        pg.draw.rect(self.window, BORDER, self.board.inflate(2, 2), 1)
        array = arena.rgb(self.team_colors, self.view)
        key = (arena.mode, self.view, self.team_colors, arena.size)
        array = self.blend.reset(array, key=key) if self.reduced_motion else self.blend.update(array, dt, key=key)
        self.window.blit(self.board_surface(array), self.board)
        self.draw_effects()
        if arena.lesson:
            self.draw_lesson_cue()
        for corner_x, corner_y, dx, dy in ((self.board.left - 2, self.board.top - 2, 1, 1),
                (self.board.right + 1, self.board.top - 2, -1, 1),
                (self.board.left - 2, self.board.bottom + 1, 1, -1),
                (self.board.right + 1, self.board.bottom + 1, -1, -1)):
            pg.draw.line(self.window, MUTED, (corner_x, corner_y), (corner_x + dx * 7, corner_y), 1)
            pg.draw.line(self.window, MUTED, (corner_x, corner_y), (corner_x, corner_y + dy * 7), 1)
        self.draw_pointer_preview()
        # Inset labels identify the field without a rounded dashboard badge.
        label = (f"LESSON / {arena.lesson.phase.upper()}" if arena.lesson else self.duel_headline() if arena.duel else
                 "PAUSED / N TO STEP" if self.paused else f"{self.view.upper()} / {arena.mode.upper()}")
        self.text(label, self.board.x + 12, self.board.y + 11, 11, AMBER if self.paused else MUTED)
        size_label = f"{arena.size} x {arena.size}"
        self.text(size_label, self.board.right - self.fonts[11].size(size_label)[0] - 12,
                  self.board.y + 11, 11, MUTED)
        # One continuous instrument rail, separated by rules rather than cards.
        pg.draw.rect(self.window, PANEL, (sx, 100, sidebar, min(640, h - 120)))
        pg.draw.line(self.window, (81, 89, 90), (sx, 100), (sx + sidebar, 100))
        self.draw_score(w)
        self.draw_sidebar(sx, 308, sidebar)
        self.draw_duel()
        self.draw_underboard()
        pg.display.flip()
        self.lesson_gate_pending = False

    def draw_underboard(self):
        x, y, width = self.board.x, self.board.bottom + 10, self.board.width
        if self.arena.lesson:
            instruction = self.lesson_instruction()
            lesson = self.lesson_snapshot
            text = (self.notices[-1][0] if self.notices and self.lesson_notice_phase == lesson["phase"] else
                    f"Count recovered after {lesson['recovery_ticks']} ticks: {lesson['baseline']} before / {lesson['remaining_after_cut']} cut / {lesson['current']} now."
                    if lesson["phase"] == "complete" else instruction[0])
            self.text(text, x, y, 12, TEXT, width)
            self.text(self.loaded_checkpoint_summary() + " / " + instruction[1] + " / G: exit",
                      x, y + 22, 11, MUTED if self.arena.left.get("health") == "ready" else CORAL, width)
            return
        if self.notices:
            text, color, _ = self.notices[-1]
            self.text(text, x, y, 12, color, width)
        else:
            no_life = not self.stats_cache["left_alive"] and not self.stats_cache["right_alive"]
            status = ("Round frozen. RESULT shows the score; R / Enter replays this seed." if self.duel_snapshot and self.duel_snapshot["finished"] else
                      self.summary["status"] if no_life or not self.stats_cache["finite"] else self.message)
            self.text(status, x, y, 12, TEXT, width)
        st = self.stats_cache
        neutral = self.arena.size ** 2 - (st["left_territory"] or 0) - (st["right_territory"] or 0)
        text = (f"{neutral:,} neutral / {st['contested']} overlap" if self.arena.mode == "hard" else
                f"{st['contested']} overlap / independent growth")
        text = f"{self.loaded_checkpoint_summary()} / {text} / {self.arena.args.device.upper()}"
        healthy = all(bundle.get("health") == "ready" for bundle in (self.arena.left, self.arena.right))
        self.text(text, x, y + 22, 11, MUTED if healthy else CORAL, width)

    def draw_sidebar(self, sx, y, sidebar):
        pg, arena = self.pg, self.arena
        self.refresh_picker_status()
        selected_side = 0 if arena.lesson else self.side
        pg.draw.line(self.window, BORDER, (sx + 12, y - 9), (sx + sidebar - 12, y - 9))
        self.text("CULTURES", sx + 12, y, 12, AMBER)
        pin, allow_unhealthy = self.picker_policy()
        policy = "CHECKPOINT AUTO" if pin is None else f"CHECKPOINT PIN {pin:03d}"
        self.text(policy, sx + sidebar - 12 - self.fonts[11].size(policy)[0], y + 1, 11, MUTED)
        if arena.lesson:
            self.text("SOLO CULTURE / 1–9 RESTARTS", sx + 12, y + 27, 11, MUTED)
        else:
            self.text("SELECT FOR", sx + 12, y + 27, 11, MUTED)
            self.button((sx + 99, y + 22, 76, 25), "LEFT", "left", self.side == 0, GREEN)
            self.button((sx + 181, y + 22, 93, 25), "RIGHT", "right", self.side == 1, CORAL)
        y += 58
        for i, (target, status) in enumerate(zip(arena.targets, self.status)):
            row, col = divmod(i, 3)
            r = pg.Rect(sx + 10 + col * 90, y + row * 62, 86, 57)
            ready = status["status"] == "ready"
            active = i == (arena.left_index if selected_side == 0 else arena.right_index)
            hover = r.collidepoint(pg.mouse.get_pos())
            fill = (53, 63, 61) if active else ((48, 57, 60) if hover else (27, 33, 35))
            pg.draw.rect(self.window, fill, r)
            if active:
                pg.draw.rect(self.window, GREEN if selected_side == 0 else CORAL, r, 1)
            thumb = self.thumbs[i]
            thumb.set_alpha(255 if ready else 60)
            self.window.blit(thumb, (r.x + 29, r.y + 2))
            self.text(str(i + 1), r.x + 5, r.y + 3, 11, MUTED)
            name = target.stem.split('_', 1)[-1]
            label = name if ready else name + " *"
            name_surface = self.fonts[13].render(label, True, TEXT if ready else MUTED)
            self.window.blit(name_surface, name_surface.get_rect(center=(r.centerx, r.y + 42)))
            self.buttons.append((r, ("target", i)))
        y += 188
        self.text("* failed / missing (research mode)" if allow_unhealthy else
                  "* failed weights / unavailable", sx + 12, y, 11, MUTED)
        y += 26
        if arena.lesson:
            self.text("GUIDED SOLO EXPERIMENT", sx + 12, y + 7, 12, AMBER)
        else:
            for dx, label, view in ((0, "LIFE", "organisms"), (90, "LAND", "territory"), (180, "PRESSURE", "pressure")):
                self.button((sx + 10 + dx, y, 86, 27), label, ("view", view), self.view == view)
        y += 42
        if not arena.duel and not arena.lesson:
            self.text("Held land leads; dead land turns neutral." if arena.mode == "hard" else
                      "Living cells lead; growth is independent.", sx + 12, y, 11, MUTED, 262)
            self.text("LEFT-CLICK TOOL", sx + 12, y + 19, 11, AMBER)
            for dx, width, label, tool, color in (
                    (10, 70, "DAMAGE", "damage", AMBER),
                    (84, 90, "PLANT LEFT", "plant_left", GREEN),
                    (178, 98, "PLANT RIGHT", "plant_right", CORAL)):
                self.button((sx + dx, y + 36, width, 27), label, ("tool", tool),
                            self.pointer_tool == tool, color)
            self.button((sx + 10, y + 69, 26, 27), "−", ("radius", -1))
            self.text(f"CUT RADIUS {self.radius}", sx + 42, y + 76, 11, TEXT, 88)
            self.button((sx + 134, y + 69, 26, 27), "+", ("radius", 1))
            self.button((sx + 176, y + 69, 100, 27), "CLEAR / C", "clear")
            return
        rules = (["Held land sets the lead. Pressure", "claims it; dead land turns neutral."] if arena.mode == "hard" else
                 ["Living cells set the comparison.", "Independent growth; no land capture."])
        if arena.duel:
            rules = ["Each scored tick adds held cells." if arena.mode == "hard" else "Scored ticks add living cells.",
                     f"First {arena.duel_config.warmup_ticks} ticks: growth warmup."]
        elif arena.lesson:
            rules = self.lesson_instruction()
        for i, text in enumerate(rules):
            self.text(text, sx + 12, y + i * 18, 13, MUTED, 262)
        help_lines = (["R: restart / G: exit lesson", "Space: pause / N: single tick", "Target: living-cell count"] if arena.lesson else
                      ["Edits locked / D: return to lab", "Cultures start a new round.", "Space: pause / N: step / R: rematch"])
        for i, line in enumerate(help_lines):
            self.text(line, sx + 12, y + 46 + i * 17, 13, TEXT, 262)

    def action(self, action):
        arena = self.arena
        self.refresh_picker_status()
        reset = False
        if action == "lesson":
            if arena.lesson:
                warning = arena.stop_lesson()
                self.pointer_tool = "damage"
                if self.lesson_prior_speed is not None:
                    self.speed = self.lesson_prior_speed
                self.lesson_prior_speed = None
                self.paused = False
                self.message = warning or "Back in a fresh lab. Plant, damage, and explore freely."
            else:
                self.lesson_prior_speed = self.speed
                arena.start_lesson()
                self.speed, self.side = 1, 0
                self.paused = True
                self.message = "A guided 48 × 48 solo experiment: plant, cut, and observe."
            self.view = "organisms"
            self.refresh(reset=True)
            return
        if action == "lesson_primary":
            self.lesson_primary()
            return
        if isinstance(action, tuple):
            if action[0] in ("tool", "radius"):
                if arena.lesson or arena.duel:
                    self.notice("Lab tools are locked. Return to LAB for planting and damage.", AMBER)
                    return
                if action[0] == "tool":
                    if action[1] in ("damage", "plant_left", "plant_right"):
                        self.pointer_tool = action[1]
                        if self.pointer_tool == "damage":
                            self.notice("Left-click damages. Shift-click: left seed; right-click: right seed.", AMBER)
                        else:
                            left = self.pointer_tool == "plant_left"
                            bundle = arena.left if left else arena.right
                            name = bundle["name"].split("_", 1)[-1].upper()
                            self.notice(f"Click plants {'LEFT' if left else 'RIGHT'} / {name}. Shift-click: L; right-click: R.",
                                        GREEN if left else CORAL)
                else:
                    self.change_radius(action[1])
                    self.notice(f"Cut radius {self.radius}: affects damage only. Planting stays one seed.", AMBER)
                return
            if action[0] == "view":
                if arena.mode == "soft" and action[1] != "organisms":
                    self.notice("Land and pressure views are available in hard mode.")
                else:
                    self.view = action[1]
                    self.notice({"organisms": "LIFE: living tissue. Turn COLORS on to identify each side.",
                                 "territory": "LAND: green is left, coral is right, dark is neutral.",
                                 "pressure": "PRESSURE: green favors left; coral favors right. Brightness shows strength."}[self.view])
            else:
                try:
                    side = 0 if arena.lesson else self.side
                    arena.select(action[1], side)
                    if arena.duel:
                        self.paused = False
                    self.refresh_picker_status(force=True)
                    self.refresh(reset=True)
                    bundle = arena.left if side == 0 else arena.right
                    name = bundle["name"].split("_", 1)[-1]
                    self.notice(f"{'Left' if side == 0 else 'Right'} {name}: {checkpoint_caption(bundle)}. Simulation seed replayed.",
                                TEXT if bundle.get("health") == "ready" else CORAL)
                except (ValueError, RuntimeError, OSError) as exc:
                    self.refresh_picker_status(force=True)
                    info = self.status[action[1]]
                    status = info["status"]
                    if not info.get("selectable", status == "ready"):
                        name = arena.targets[action[1]].stem.split("_", 1)[-1].title()
                        pin, _ = self.picker_policy()
                        selection = "checkpoint" if pin is None else f"seed {pin:03d}"
                        self.notice(f"{name}: {status} {selection}. Choose a ready culture.", AMBER)
                    else:
                        self.notice(str(exc), CORAL)
            self.refresh()
            return
        if action == "pause":
            if arena.lesson:
                if arena.lesson.phase == "injured":
                    self.lesson_primary()
                    return
                if not arena.lesson.can_step:
                    self.notice(" ".join(self.lesson_instruction()), AMBER)
                    return
            if arena.duel and arena.duel.finished:
                self.notice("Round complete. R replays this seed; NEXT SEED starts another.", AMBER)
                return
            self.paused = not self.paused
            self.lesson_credit = 0.0
            self.notice(("Paused. N advances one tick; guided edits stay locked." if arena.lesson else
                         "Paused. N advances one tick; duel edits stay locked." if arena.duel else
                         "Paused. Editing still works; N advances one tick.") if self.paused else "Simulation resumed.")
        elif action == "reset":
            arena.reset()
            if arena.duel:
                self.paused = False
            reset = True
            self.message = "Replaying the same seed."
        elif action == "duel":
            if arena.lesson and self.lesson_prior_speed is not None:
                self.speed = self.lesson_prior_speed
                self.lesson_prior_speed = None
            warning = arena.set_duel(not arena.duel_enabled)
            if not arena.duel:
                self.pointer_tool = "damage"
            self.paused, reset = False, True
            self.message = ("Duel started. Hold more land over time; edits are locked." if arena.mode == "hard" else
                            "Growth duel started. Sustain more living cells; edits are locked.") if arena.duel else \
                           "Back in the lab. Plant, damage, and explore freely."
            if warning:
                self.message = warning
        elif action == "next_seed":
            if not arena.duel or not arena.duel.finished:
                return
            arena.args.seed = (arena.args.seed + 1) % (2 ** 32)
            arena.reset()
            self.paused, reset = False, True
            self.message = f"New duel / seed {arena.args.seed}."
        elif action == "result":
            if arena.duel and arena.duel.finished:
                self.result_visible = not self.result_visible
                if not self.result_visible and self.result_rect:
                    self.stale_result_rect = self.result_rect.copy()
        elif action == "save_duel":
            self.save_duel()
            return
        elif action == "step":
            if arena.lesson and arena.lesson.phase == "injured":
                if self.lesson_gate_pending:
                    return
                arena.watch_lesson()
                self.lesson_gate_pending = True
            self.paused = True
            self.lesson_credit = 0.0
            advanced = arena.step()
            self.notice("Advanced exactly one simulation tick." if advanced else
                        " ".join(self.lesson_instruction()) if arena.lesson else "Round complete. R starts a rematch.")
        elif action == "speed":
            self.speed = 1 if self.speed == 8 else self.speed * 2
            if arena.lesson and arena.lesson.phase == "recover":
                self.notice(f"Slow observation: {6 * self.speed} simulation ticks per second. N inspects one tick.")
        elif action == "mode":
            if arena.lesson:
                self.notice("This lesson uses solo soft growth. G returns to the lab's rules.", AMBER)
                return
            arena.mode = "soft" if arena.mode == "hard" else "hard"
            arena.reset()
            if arena.duel:
                self.paused = False
            self.view, reset = "organisms", True
            self.message = f"{arena.mode.title()} rules. Replaying the same seed."
        elif action == "clear":
            if arena.lesson:
                self.notice("R restarts the lesson; G returns to free planting and damage.", AMBER)
                return
            if arena.duel:
                self.notice("Duel edits are locked. BACK TO LAB restores planting and damage.", AMBER)
                return
            arena.clear()
            reset = True
            self.message = "Arena cleared. Choose PLANT LEFT / PLANT RIGHT, then click the field."
        elif action == "colors":
            self.team_colors = not self.team_colors
        elif action == "motion":
            self.reduced_motion = not self.reduced_motion
        elif action in ("left", "right"):
            self.side = 0 if arena.lesson or action == "left" else 1
        self.refresh(reset=reset)
        if action == "clear":
            self.notice(self.message)

    def event(self, event):
        pg, arena = self.pg, self.arena
        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
            return False
        # Process modifiers in event order. Current physical polling can see
        # either a later press or release from this same drained SDL batch.
        if event.type == pg.KEYDOWN and event.key in (pg.K_LSHIFT, pg.K_RSHIFT):
            self.held_shift_keys.add(event.key)
        elif event.type == pg.KEYUP and event.key in (pg.K_LSHIFT, pg.K_RSHIFT):
            self.held_shift_keys.discard(event.key)
        elif event.type == pg.WINDOWFOCUSLOST:
            self.held_shift_keys.clear()
        if event.type == pg.VIDEORESIZE:
            self.window = pg.display.set_mode((max(860, event.w), max(740, event.h)), pg.RESIZABLE)
            self.draw(dt=0)  # Subsequent clicks in this event batch use current geometry.
        if event.type == pg.KEYDOWN:
            mapping = {pg.K_SPACE: "pause", pg.K_r: "reset", pg.K_n: "step", pg.K_c: "clear",
                       pg.K_t: "colors", pg.K_m: "mode", pg.K_TAB: "speed", pg.K_f: "motion",
                       pg.K_d: "duel", pg.K_g: "lesson", pg.K_s: "save_duel"}
            if event.key == pg.K_RETURN:
                if arena.lesson:
                    self.action("lesson_primary")
                elif arena.duel and arena.duel.finished:
                    self.action("reset")
            if event.key in mapping:
                self.action(mapping[event.key])
            elif event.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                if arena.lesson:
                    self.notice("The lesson previews an exact safe cut. G restores the free damage brush.", AMBER)
                else:
                    self.change_radius(1 if event.key == pg.K_RIGHTBRACKET else -1)
            elif pg.K_1 <= event.key <= pg.K_9:
                index = event.key - pg.K_1
                if index < len(arena.targets):
                    self.side = 0 if arena.lesson else 1 if event.mod & pg.KMOD_SHIFT else 0
                    self.action(("target", index))
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.stale_result_rect and self.stale_result_rect.collidepoint(event.pos):
                return True  # Never click through a dismissed result in this event batch.
            if event.button == 1:
                controls = (self.result_buttons + self.buttons
                            if arena.duel and arena.duel.finished and self.result_visible else self.buttons)
                for rect, action in controls:
                    if rect.collidepoint(event.pos):
                        old_result = self.result_rect.copy() if self.result_rect and self.result_visible else None
                        self.action(action)
                        if old_result and (not arena.duel or not arena.duel.finished or not self.result_visible):
                            self.stale_result_rect = old_result
                        return True
            if self.board.collidepoint(event.pos):
                x, y = self.pointer_cell(event.pos)
                tool = self.effective_pointer_tool(event.button)
                if tool == "damage":
                    self.interact(x, y)
                elif tool in ("plant_left", "plant_right"):
                    self.interact(x, y, 0 if tool == "plant_left" else 1)
        return True

    def run(self):
        frames, running = 0, True
        try:
            self.draw()
            while running:
                for event in self.pg.event.get():
                    if not self.event(event):
                        running = False
                        break
                if not running:
                    break
                budget = self.step_budget(self.clock.get_time() / 1000.0)
                if budget:
                    self.arena.step(budget)
                self.current_fps = self.clock.get_fps()
                self.draw()
                frames += 1
                if self.arena.args.ui_frames and frames >= self.arena.args.ui_frames:
                    break
                self.clock.tick(self.arena.args.fps)
            if self.arena.args.snapshot:
                # Quit may follow an edit/step in the same event batch. Export
                # the final state and HUD, without an older interpolated frame.
                self.blend.reset()
                self.draw(dt=0)
                self.arena.args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                self.pg.image.save(self.window, str(self.arena.args.snapshot))
        finally:
            self.pg.quit()


def main(argv=None, default_mode="hard"):
    args = parse_args(argv, default_mode)
    targets = list_targets()
    if args.list_models:
        for i, target in enumerate(targets, 1):
            status = target_status(target)
            score = f" score={status['score']:.6g}" if status["score"] is not None else ""
            print(f"{i}: {target.stem:14} {status['status']:10} seed={status['seed']}{score}")
        return
    if not targets or not (1 <= args.left <= len(targets) and 1 <= args.right <= len(targets)):
        raise SystemExit(f"--left and --right must be between 1 and {len(targets)}; see --list-models")
    args.device = pick_device(args.device)
    try:
        configure_runtime(seed=args.seed, cpu_threads=args.cpu_threads)
        arena = Arena(args, targets)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    started = time.perf_counter()
    if args.headless_frames:
        arena.step(args.headless_frames)
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arena.rgb(args.team_colors)).resize((768, 768), Image.Resampling.NEAREST).save(args.snapshot)
    else:
        ArenaUI(arena).run()
    elapsed = time.perf_counter() - started
    report = arena.report(elapsed)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["duel"] and not report["duel"]["valid"]:
        raise SystemExit(report["duel"]["invalid_reason"])
    if report["lesson"] and not report["lesson"]["valid"]:
        raise SystemExit(report["lesson"]["reason"])
    if not report["finite"]:
        raise SystemExit("Simulation produced non-finite state; check the selected checkpoints.")
    if args.export_duel:
        from duel_recipe import write_recipe
        try:
            write_recipe(args.export_duel, arena.duel_recipe())
        except (ValueError, OSError, RuntimeError) as exc:
            raise SystemExit(f"Could not export duel: {exc}") from exc
        print(f"Saved duel recipe: {args.export_duel}", file=sys.stderr)


if __name__ == "__main__":
    main()
