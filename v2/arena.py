"""Interactive arena and genuinely render-free, reproducible simulation runner."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from battle import clash_step, compose_rgba, crater
from clash import (V2_ROOT, active_grid_size, clamp, ensure_model, list_targets,
                   parse_grid_pos, reset_world, target_status)
from runtime import configure_runtime
from trainer.common import pick_device


class Arena:
    def __init__(self, args, targets):
        self.args, self.targets = args, targets
        self.cache = {}
        self.left_index, self.right_index = args.left - 1, args.right - 1
        self.left = self.load(self.left_index, args.left_seed)
        self.right = self.load(self.right_index, args.right_seed)
        self.mode = args.mode
        self.steps = 0
        self.reset()

    def load(self, index, seed=None):
        key = index, seed
        if key not in self.cache:
            self.cache[key] = ensure_model(self.targets[index], self.args.device,
                self.args.bootstrap_steps, preferred_seed=seed,
                allow_unhealthy=self.args.allow_unhealthy)
        return self.cache[key]

    def reset(self):
        # Reset is a replay of this seed. Model loading must not perturb the match RNG.
        configure_runtime(seed=self.args.seed, cpu_threads=self.args.cpu_threads)
        self.size = active_grid_size(self.args.grid_size, self.left, self.right)
        self.a, self.b, self.owner, self.control = reset_world(
            self.size, self.args.device, self.left, self.right,
            left_pos=self.args.left_pos, right_pos=self.args.right_pos)
        self.steps = 0

    def select(self, index, side):
        bundle = self.load(index, self.args.left_seed if side == 0 else self.args.right_seed)
        if side == 0:
            self.left, self.left_index = bundle, index
        else:
            self.right, self.right_index = bundle, index
        self.reset()

    def clear(self):
        self.a = torch.zeros_like(self.a)
        self.b = torch.zeros_like(self.b)
        self.owner = torch.zeros_like(self.owner)
        self.control = torch.zeros_like(self.control)
        self.steps = 0

    def damage(self, x, y, radius):
        self.a, self.b, self.owner, self.control = crater(
            self.a, self.b, self.owner, self.control, x, y, radius)

    def plant(self, x, y, side):
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
        with torch.inference_mode():
            for _ in range(count):
                if self.mode == "hard":
                    self.a, self.b, self.owner, self.control = clash_step(
                        self.a, self.b, self.owner, self.control,
                        self.left["model"], self.right["model"],
                        pressure_gain=self.args.pressure_gain,
                        control_decay=self.args.control_decay,
                        capture_threshold=self.args.capture_threshold,
                        release_threshold=self.args.release_threshold,
                        tie_margin=self.args.tie_margin)
                else:
                    self.a = self.left["model"](self.a)
                    self.b = self.right["model"](self.b)
                self.steps += 1

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
                "right": {k: self.right.get(k) for k in ("name", "source", "seed_dir", "score", "health")},
                "rules": {k: getattr(self.args, k) for k in ("pressure_gain", "control_decay",
                    "capture_threshold", "release_threshold", "tie_margin")}}


def parse_args(argv=None, default_mode="hard"):
    parser = argparse.ArgumentParser(description="Petri Clash: a living neural arena. No training required.")
    parser.add_argument("--mode", choices=("hard", "soft"), default=default_mode)
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
    parser.add_argument("--headless-frames", type=int, default=0, help="simulate exactly N steps with no SDL, rendering, or FPS sleep")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--report", type=Path, help="write final machine-readable match report")
    parser.add_argument("--snapshot", type=Path, help="save final PNG (board headlessly; entire interactive window)")
    parser.add_argument("--ui-frames", type=int, default=0, help="exit interactive mode after N frames, useful for UI smoke tests")
    args = parser.parse_args(argv)
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
        self.status = [target_status(t) for t in arena.targets]
        self.thumbs = [pygame.image.load(str(t)).convert_alpha() for t in arena.targets]
        self.paused, self.side, self.view = False, 0, "organisms"
        self.speed, self.radius = arena.args.steps_per_frame, arena.args.crater_radius
        self.team_colors = arena.args.team_colors
        self.reduced_motion = arena.args.reduced_motion
        self.message = "Grow. Collide. Regenerate."
        self.buttons = []
        self.board = pygame.Rect(0, 0, 1, 1)
        self.clock = pygame.time.Clock()
        self.stats_cache = arena.stats()
        self.current_fps = 0.0
        self.blend = FrameBlend()
        self.score = ScoreTracker()
        self.effects = []
        self.notices = []
        self.last_draw = time.perf_counter()
        self.elapsed = 0.0
        self.refresh(reset=True)

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
        self.stats_cache = self.arena.stats()
        if reset:
            from feedback import FrameBlend
            self.blend = FrameBlend()
            self.score.reset()
            self.effects.clear()
            self.notices.clear()
        self.summary = self.score.update(self.stats_cache)

    def notice(self, text, color=TEXT):
        self.message = text
        self.notices.append((text, color, self.elapsed))
        self.notices = self.notices[-3:]

    def interact(self, x, y, side=None):
        """Apply one input, then report its real immediate effect, even paused."""
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
        pg, st = self.pg, self.stats_cache
        sx, y = self.rail_x, 111
        self.score_rect = pg.Rect(sx, y, 286, 181)
        hard = self.arena.mode == "hard"
        left, right = self.summary["left"], self.summary["right"]
        self.text("HELD CELLS" if hard else "LIVING CELLS", sx + 12, y, 12, AMBER)
        self.text("LIVE" if not self.paused else "PAUSED", sx + 222, y, 11, MUTED)
        for side, value, color in ((0, left, GREEN), (1, right, CORAL)):
            x = sx + 12 + side * 140
            name = (self.arena.left if side == 0 else self.arena.right)["name"].split("_", 1)[-1]
            self.text(f"{'L' if side == 0 else 'R'} / {name.upper()}", x, y + 24, 12, color, 126)
            size = 36 if self.fonts[36].size(f"{value:,}")[0] <= 126 else 28
            self.text(f"{value:,}", x - 2, y + 40, size, color, 128)
            alive = st["left_alive" if side == 0 else "right_alive"]
            pct = self.summary["percent"]["left" if side == 0 else "right"]
            self.text(f"{pct:.1f}% held" if hard else f"{pct:.1f}% alive", x, y + 83, 13, MUTED)
            self.text(f"{alive} living cells", x, y + 100, 13, MUTED, 128)
        self.text(self.summary["headline"], sx + 12, y + 124, 16, TEXT, 262)
        self.text(self.summary["trend"] if self.summary["span"] else "Open-ended / no final winner", sx + 12, y + 146, 11, MUTED, 262)
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

    def draw_effects(self):
        pg, arena = self.pg, self.arena
        scale = self.board.width / arena.size
        overlay = pg.Surface(self.board.size, pg.SRCALPHA)
        self.effects = [effect for effect in self.effects if self.elapsed - effect["born"] < 1.25]
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
        sidebar, by, gap = 286, 110, 16
        size = max(1, min(w - sidebar - 56, h - by - 52))
        bx = max(20, (w - size - gap - sidebar) // 2)
        sx = bx + size + gap
        self.rail_x = sx
        self.board = pg.Rect(bx, by, size, size)
        self.text("PETRI CLASH", bx, 13, 28)
        self.text("NEURAL GROWTH SANDBOX", bx + 1, 49, 11, MUTED)
        self.text(f"SEED {arena.args.seed:03d}", sx + 12, 23, 12, AMBER)
        self.text(f"TICK {arena.steps:05d} / {self.current_fps:.0f} FPS", sx + 12, 47, 11, MUTED)
        x = bx
        for width, label, action, active in (
            (74, "RESUME" if self.paused else "PAUSE", "pause", self.paused),
            (68, "REPLAY", "reset", False), (48, "STEP", "step", False),
            (42, f"{self.speed}x", "speed", False),
            (66, "HARD" if arena.mode == "hard" else "SOFT", "mode", True),
            (72, "COLORS", "colors", self.team_colors),
            (60, "FX OFF" if self.reduced_motion else "FX ON", "motion", not self.reduced_motion)):
            self.button((x, 73, width, 27), label, action, active)
            x += width + 6
        # The board is the dominant element; the rail always sits 16px beside it.
        pg.draw.rect(self.window, BORDER, self.board.inflate(2, 2), 1)
        array = arena.rgb(self.team_colors, self.view)
        key = (arena.mode, self.view, self.team_colors, arena.size)
        array = self.blend.reset(array, key=key) if self.reduced_motion else self.blend.update(array, dt, key=key)
        surface = pg.surfarray.make_surface(array.swapaxes(0, 1))
        self.window.blit(pg.transform.scale(surface, self.board.size), self.board)
        self.draw_effects()
        for corner_x, corner_y, dx, dy in ((self.board.left - 2, self.board.top - 2, 1, 1),
                (self.board.right + 1, self.board.top - 2, -1, 1),
                (self.board.left - 2, self.board.bottom + 1, 1, -1),
                (self.board.right + 1, self.board.bottom + 1, -1, -1)):
            pg.draw.line(self.window, MUTED, (corner_x, corner_y), (corner_x + dx * 7, corner_y), 1)
            pg.draw.line(self.window, MUTED, (corner_x, corner_y), (corner_x, corner_y + dy * 7), 1)
        mouse = pg.mouse.get_pos()
        if self.board.collidepoint(mouse):
            planting = bool(pg.key.get_mods() & pg.KMOD_SHIFT)
            radius = max(3, round((1 if planting else self.radius) * self.board.width / arena.size))
            pg.draw.circle(self.window, GREEN if planting else AMBER, mouse, radius, 1)
        # Inset labels identify the field without a rounded dashboard badge.
        label = "PAUSED / N TO STEP" if self.paused else f"{self.view.upper()} / {arena.mode.upper()}"
        self.text(label, self.board.x + 12, self.board.y + 11, 11, AMBER if self.paused else MUTED)
        size_label = f"{arena.size} x {arena.size}"
        self.text(size_label, self.board.right - self.fonts[11].size(size_label)[0] - 12,
                  self.board.y + 11, 11, MUTED)
        # One continuous instrument rail, separated by rules rather than cards.
        pg.draw.rect(self.window, PANEL, (sx, 100, sidebar, min(640, h - 120)))
        pg.draw.line(self.window, (81, 89, 90), (sx, 100), (sx + sidebar, 100))
        self.draw_score(w)
        self.draw_sidebar(sx, 308, sidebar)
        self.draw_underboard()
        pg.display.flip()

    def draw_underboard(self):
        x, y, width = self.board.x, self.board.bottom + 10, self.board.width
        if self.notices:
            text, color, _ = self.notices[-1]
            self.text(text, x, y, 12, color, width)
        else:
            no_life = not self.stats_cache["left_alive"] and not self.stats_cache["right_alive"]
            status = self.summary["status"] if no_life or not self.stats_cache["finite"] else self.message
            self.text(status, x, y, 12, TEXT, width)
        st = self.stats_cache
        neutral = self.arena.size ** 2 - (st["left_territory"] or 0) - (st["right_territory"] or 0)
        text = f"{neutral:,} neutral / {st['contested']} overlapping / {self.arena.args.device.upper()}" if self.arena.mode == "hard" else f"{st['contested']} overlapping / independent growth / {self.arena.args.device.upper()}"
        self.text(text, x, y + 22, 11, MUTED, width)

    def draw_sidebar(self, sx, y, sidebar):
        pg, arena = self.pg, self.arena
        pg.draw.line(self.window, BORDER, (sx + 12, y - 9), (sx + sidebar - 12, y - 9))
        self.text("CULTURES", sx + 12, y, 12, AMBER)
        self.text("SELECT FOR", sx + 12, y + 27, 11, MUTED)
        self.button((sx + 99, y + 22, 76, 25), "LEFT", "left", self.side == 0, GREEN)
        self.button((sx + 181, y + 22, 93, 25), "RIGHT", "right", self.side == 1, CORAL)
        y += 58
        for i, (target, status) in enumerate(zip(arena.targets, self.status)):
            row, col = divmod(i, 3)
            r = pg.Rect(sx + 10 + col * 90, y + row * 62, 86, 57)
            ready = status["status"] == "ready"
            active = i == (arena.left_index if self.side == 0 else arena.right_index)
            hover = r.collidepoint(pg.mouse.get_pos())
            fill = (53, 63, 61) if active else ((48, 57, 60) if hover else (27, 33, 35))
            pg.draw.rect(self.window, fill, r)
            if active:
                pg.draw.rect(self.window, GREEN if self.side == 0 else CORAL, r, 1)
            thumb = pg.transform.smoothscale(self.thumbs[i], (28, 28))
            if not ready:
                thumb.set_alpha(60)
            self.window.blit(thumb, (r.x + 29, r.y + 2))
            self.text(str(i + 1), r.x + 5, r.y + 3, 11, MUTED)
            name = target.stem.split('_', 1)[-1]
            label = name if ready else name + " *"
            name_surface = self.fonts[13].render(label, True, TEXT if ready else MUTED)
            self.window.blit(name_surface, name_surface.get_rect(center=(r.centerx, r.y + 42)))
            self.buttons.append((r, ("target", i)))
        y += 188
        self.text("* failed weights / unavailable", sx + 12, y, 11, MUTED)
        y += 26
        for dx, label, view in ((0, "LIFE", "organisms"), (90, "LAND", "territory"), (180, "PRESSURE", "pressure")):
            self.button((sx + 10 + dx, y, 86, 27), label, ("view", view), self.view == view)
        y += 42
        rules = (["Held land sets the lead. Pressure", "claims it; dead land turns neutral."] if arena.mode == "hard" else
                 ["Living cells set the comparison.", "Independent growth; no land capture."])
        for i, text in enumerate(rules):
            self.text(text, sx + 12, y + i * 18, 13, MUTED, 262)
        self.text(f"Click: damage [{self.radius}]  /  [ ]: size", sx + 12, y + 46, 13, TEXT, 262)
        self.text("Shift-click: L seed / Right: R seed", sx + 12, y + 63, 13, TEXT, 262)
        self.text("Space: pause / N: step / C: clear", sx + 12, y + 80, 13, TEXT, 262)

    def action(self, action):
        arena = self.arena
        reset = False
        if isinstance(action, tuple):
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
                    arena.select(action[1], self.side)
                    self.refresh(reset=True)
                    self.notice("Culture loaded. Replaying the same seed.")
                except (ValueError, RuntimeError, OSError) as exc:
                    status = self.status[action[1]]["status"]
                    if status != "ready":
                        name = arena.targets[action[1]].stem.split("_", 1)[-1].title()
                        self.notice(f"{name}: {status} checkpoint. Choose a ready culture.")
                    else:
                        self.notice(str(exc))
            self.refresh()
            return
        if action == "pause":
            self.paused = not self.paused
            self.notice("Paused. Editing still works; N advances one tick." if self.paused else "Simulation resumed.")
        elif action == "reset":
            arena.reset()
            reset = True
            self.message = "Replaying the same seed."
        elif action == "step":
            self.paused = True
            arena.step()
            self.notice("Advanced exactly one simulation tick.")
        elif action == "speed":
            self.speed = 1 if self.speed == 8 else self.speed * 2
        elif action == "mode":
            arena.mode = "soft" if arena.mode == "hard" else "hard"
            arena.reset()
            self.view, reset = "organisms", True
            self.message = f"{arena.mode.title()} rules. Replaying the same seed."
        elif action == "clear":
            arena.clear()
            reset = True
            self.message = "Arena cleared. Shift-click or right-click to plant new life."
        elif action == "colors":
            self.team_colors = not self.team_colors
        elif action == "motion":
            self.reduced_motion = not self.reduced_motion
        elif action in ("left", "right"):
            self.side = 0 if action == "left" else 1
        self.refresh(reset=reset)

    def event(self, event):
        pg, arena = self.pg, self.arena
        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
            return False
        if event.type == pg.VIDEORESIZE:
            self.window = pg.display.set_mode((max(860, event.w), max(740, event.h)), pg.RESIZABLE)
            self.draw(dt=0)  # Subsequent clicks in this event batch use current geometry.
        if event.type == pg.KEYDOWN:
            mapping = {pg.K_SPACE: "pause", pg.K_r: "reset", pg.K_n: "step", pg.K_c: "clear",
                       pg.K_t: "colors", pg.K_m: "mode", pg.K_TAB: "speed", pg.K_f: "motion"}
            if event.key in mapping:
                self.action(mapping[event.key])
            elif event.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                self.radius = clamp(self.radius + (1 if event.key == pg.K_RIGHTBRACKET else -1), 1, arena.size)
            elif pg.K_1 <= event.key <= pg.K_9:
                index = event.key - pg.K_1
                if index < len(arena.targets):
                    self.side = 1 if event.mod & pg.KMOD_SHIFT else 0
                    self.action(("target", index))
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.board.collidepoint(event.pos):
                x = clamp(int((event.pos[0] - self.board.x) * arena.size / self.board.width), 0, arena.size - 1)
                y = clamp(int((event.pos[1] - self.board.y) * arena.size / self.board.height), 0, arena.size - 1)
                if event.button == 3:
                    self.interact(x, y, 1)
                elif event.button == 1 and pg.key.get_mods() & pg.KMOD_SHIFT:
                    self.interact(x, y, 0)
                elif event.button == 1:
                    self.interact(x, y)
            elif event.button == 1:
                for rect, action in self.buttons:
                    if rect.collidepoint(event.pos):
                        self.action(action)
                        break
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
                if not self.paused:
                    self.arena.step(self.speed)
                self.current_fps = self.clock.get_fps()
                self.draw()
                frames += 1
                if self.arena.args.ui_frames and frames >= self.arena.args.ui_frames:
                    break
                self.clock.tick(self.arena.args.fps)
            if self.arena.args.snapshot:
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
    if not report["finite"]:
        raise SystemExit("Simulation produced non-finite state; check the selected checkpoints.")


if __name__ == "__main__":
    main()
