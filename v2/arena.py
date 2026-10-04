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
            rgb = torch.stack((c.clamp_min(0), c.abs() * .35, (-c).clamp_min(0)))
        elif team_colors or view == "territory":
            a = self.a[0, 3].detach().cpu().clamp(0, 1)
            b = self.b[0, 3].detach().cpu().clamp(0, 1)
            if view == "territory" and self.mode == "hard":
                a = (self.owner[0, 0].detach().cpu() == 1).float()
                b = (self.owner[0, 0].detach().cpu() == 2).float()
            rgb = torch.stack((a * .30 + b * .92, a * .88 + b * .44, a * .74 + b * .55)).clamp(0, 1)
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


BG, PANEL, BORDER = (12, 19, 23), (21, 32, 37), (44, 60, 64)
TEXT, MUTED, GREEN, CORAL = (231, 238, 222), (142, 160, 159), (93, 226, 178), (246, 132, 148)


class ArenaUI:
    """Small native UI, no assets/downloads or renderer dependencies beyond pygame."""
    def __init__(self, arena):
        import pygame
        self.pg, self.arena = pygame, arena
        pygame.display.init()
        pygame.font.init()
        width = max(860, arena.args.window_size)
        self.window = pygame.display.set_mode((width, max(740, int(width * .82))), pygame.RESIZABLE)
        pygame.display.set_caption("Petri Clash | Living arena")
        self.fonts = {s: pygame.font.SysFont("DejaVu Sans", s) for s in (11, 12, 13, 14, 16, 18, 22, 28)}
        self.status = [target_status(t) for t in arena.targets]
        self.thumbs = [pygame.image.load(str(t)).convert_alpha() for t in arena.targets]
        self.paused, self.side, self.view = False, 0, "organisms"
        self.speed = arena.args.steps_per_frame
        self.radius = arena.args.crater_radius
        self.team_colors = arena.args.team_colors
        self.message = "Grow. Collide. Regenerate."
        self.buttons = []
        self.board = pygame.Rect(0, 0, 1, 1)
        self.clock = pygame.time.Clock()
        self.stats_cache = arena.stats()
        self.current_fps = 0.0

    def text(self, text, x, y, size=14, color=TEXT):
        self.window.blit(self.fonts[size].render(str(text), True, color), (x, y))

    def button(self, rect, label, action, active=False, color=GREEN):
        pg = self.pg
        rect = pg.Rect(rect)
        hover = rect.collidepoint(pg.mouse.get_pos())
        fill = (38, 65, 60) if active else ((32, 47, 51) if hover else PANEL)
        pg.draw.rect(self.window, fill, rect, border_radius=7)
        pg.draw.rect(self.window, color if active else BORDER, rect, 1, border_radius=7)
        surface = self.fonts[12].render(label, True, color if active else TEXT)
        self.window.blit(surface, surface.get_rect(center=rect.center))
        self.buttons.append((rect, action))

    def draw(self):
        pg, arena = self.pg, self.arena
        w, h = self.window.get_size()
        self.window.fill(BG)
        self.buttons = []
        sidebar = 270
        bx, by = 22, 132
        size = max(1, min(w - sidebar - 64, h - by - 74))
        self.board = pg.Rect(bx, by, size, size)
        sx = w - sidebar - 22
        self.text("PETRI / CLASH", 22, 18, 28)
        self.text("LIVING NEURAL ARENA", 24, 57, 11, GREEN)
        self.text(f"{arena.args.device.upper()}  /  {torch.get_num_threads()} THREADS", sx, 25, 12, MUTED)
        self.text(f"SEED {arena.args.seed}  /  STEP {arena.steps:05d}", sx, 48, 12)
        self.button((22, 88, 85, 30), "RESUME" if self.paused else "PAUSE", "pause", self.paused)
        self.button((115, 88, 76, 30), "RESET", "reset")
        self.button((199, 88, 68, 30), "STEP", "step")
        self.button((275, 88, 68, 30), f"{self.speed}x", "speed")
        self.button((351, 88, 112, 30), "HARD" if arena.mode == "hard" else "SOFT", "mode", True)
        self.button((471, 88, 90, 30), "COLORS", "colors", self.team_colors)
        pg.draw.rect(self.window, BORDER, self.board.inflate(2, 2), border_radius=3)
        array = arena.rgb(self.team_colors, self.view)
        surface = pg.surfarray.make_surface(array.swapaxes(0, 1))
        self.window.blit(pg.transform.scale(surface, self.board.size), self.board)
        mouse = pg.mouse.get_pos()
        if self.board.collidepoint(mouse):
            radius = max(3, round(self.radius * self.board.width / arena.size))
            pg.draw.circle(self.window, (215, 229, 212), mouse, radius, 1)
        if self.paused:
            badge = pg.Rect(self.board.x + 12, self.board.y + 12, 151, 27)
            pg.draw.rect(self.window, PANEL, badge, border_radius=6)
            self.text("PAUSED / N TO STEP", badge.x + 9, badge.y + 6, 11, GREEN)
        st = self.stats_cache
        total = max(1, st["left_alive"] + st["right_alive"])
        bar = pg.Rect(bx, self.board.bottom + 13, size, 5)
        pg.draw.rect(self.window, CORAL, bar)
        pg.draw.rect(self.window, GREEN, (bar.x, bar.y, round(size * st["left_alive"] / total), 5))
        self.text(f"{st['left_alive']} living cells", bx, bar.bottom + 7, 12, GREEN)
        label = f"{st['right_alive']} living cells"
        self.text(label, self.board.right - self.fonts[12].size(label)[0], bar.bottom + 7, 12, CORAL)
        self.text(f"{self.current_fps:.0f} FPS  /  {self.view.upper()}", bx, h - 24, 11, MUTED)
        self.text(self.message[:76], bx + 165, h - 24, 11, TEXT)
        y = 90
        for side, bundle in enumerate((arena.left, arena.right)):
            color = GREEN if side == 0 else CORAL
            pg.draw.rect(self.window, PANEL, (sx, y, sidebar, 98), border_radius=10)
            self.text("LEFT CULTURE" if side == 0 else "RIGHT CULTURE", sx + 14, y + 10, 11, color)
            self.text(bundle["name"].split("_", 1)[-1].upper(), sx + 14, y + 30, 22)
            territory = st["left_territory" if side == 0 else "right_territory"]
            details = f"{bundle['seed_dir']}  /  {bundle.get('health', 'unverified')}"
            self.text(details, sx + 14, y + 60, 11, MUTED)
            if territory is not None:
                self.text(f"{territory} owned cells", sx + 14, y + 77, 11, color)
            y += 108
        self.text("CHOOSE A CULTURE", sx, y + 3, 12, MUTED)
        y += 27
        self.button((sx, y, 130, 29), "FOR LEFT", "left", self.side == 0, GREEN)
        self.button((sx + 140, y, 130, 29), "FOR RIGHT", "right", self.side == 1, CORAL)
        y += 39
        for i, (target, status) in enumerate(zip(arena.targets, self.status)):
            row, col = divmod(i, 3)
            r = pg.Rect(sx + col * 92, y + row * 74, 86, 68)
            ready = status["status"] == "ready"
            active = i == (arena.left_index if self.side == 0 else arena.right_index)
            hover = r.collidepoint(pg.mouse.get_pos())
            fill = (36, 59, 54) if active else ((32, 47, 51) if hover else PANEL)
            pg.draw.rect(self.window, fill, r, border_radius=7)
            pg.draw.rect(self.window, GREEN if active else BORDER, r, 1, border_radius=7)
            thumb = pg.transform.smoothscale(self.thumbs[i], (30, 30))
            if not ready:
                thumb.set_alpha(65)
            self.window.blit(thumb, (r.x + 28, r.y + 4))
            name = f"{i + 1} {target.stem.split('_', 1)[-1]}"
            self.text(name, r.x + 5, r.y + 35, 11, TEXT if ready else MUTED)
            self.text("ready" if ready else status["status"], r.x + 5, r.y + 51, 11, GREEN if ready else MUTED)
            self.buttons.append((r, ("target", i)))
        y += 232
        self.button((sx, y, 86, 28), "LIFE", ("view", "organisms"), self.view == "organisms")
        self.button((sx + 92, y, 86, 28), "LAND", ("view", "territory"), self.view == "territory")
        self.button((sx + 184, y, 86, 28), "PRESSURE", ("view", "pressure"), self.view == "pressure")
        self.text(f"CRATER RADIUS {self.radius}   [ / ]", sx, y + 42, 11, MUTED)
        self.text("Click: damage / Shift-click: left seed", sx, y + 62, 11, MUTED)
        self.text("Right-click: right seed / C: clear", sx, y + 79, 11, MUTED)
        self.text("Space: pause / N: step / R: replay", sx, y + 96, 11, MUTED)
        pg.display.flip()

    def action(self, action):
        arena = self.arena
        if isinstance(action, tuple):
            if action[0] == "view":
                if arena.mode == "soft" and action[1] != "organisms":
                    self.message = "Land and pressure views are available in hard mode."
                else:
                    self.view = action[1]
            else:
                try:
                    arena.select(action[1], self.side)
                    self.message = "Culture loaded. Match reset to the same seed."
                except (ValueError, RuntimeError, OSError) as exc:
                    status = self.status[action[1]]["status"]
                    if status != "ready":
                        name = arena.targets[action[1]].stem.split("_", 1)[-1].title()
                        self.message = f"{name}: {status} checkpoint. Choose a ready culture."
                    else:
                        self.message = str(exc)
            self.stats_cache = arena.stats()
            return
        if action == "pause":
            self.paused = not self.paused
        elif action == "reset":
            arena.reset()
            self.message = "Replaying the same seed."
        elif action == "step":
            self.paused = True
            arena.step()
        elif action == "speed":
            self.speed = 1 if self.speed == 8 else self.speed * 2
        elif action == "mode":
            arena.mode = "soft" if arena.mode == "hard" else "hard"
            arena.reset()
            self.view = "organisms"
            self.message = f"{arena.mode.title()} rules. Match reset."
        elif action == "colors":
            self.team_colors = not self.team_colors
        elif action in ("left", "right"):
            self.side = 0 if action == "left" else 1
        self.stats_cache = arena.stats()

    def event(self, event):
        pg, arena = self.pg, self.arena
        if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
            return False
        if event.type == pg.VIDEORESIZE:
            self.window = pg.display.set_mode((max(860, event.w), max(740, event.h)), pg.RESIZABLE)
        if event.type == pg.KEYDOWN:
            mapping = {pg.K_SPACE: "pause", pg.K_r: "reset", pg.K_n: "step",
                       pg.K_t: "colors", pg.K_m: "mode", pg.K_TAB: "speed"}
            if event.key in mapping:
                self.action(mapping[event.key])
            elif event.key == pg.K_c:
                arena.clear()
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
                    arena.plant(x, y, 1)
                elif event.button == 1 and pg.key.get_mods() & pg.KMOD_SHIFT:
                    arena.plant(x, y, 0)
                elif event.button == 1:
                    arena.damage(x, y, self.radius)
            elif event.button == 1:
                for rect, action in self.buttons:
                    if rect.collidepoint(event.pos):
                        self.action(action)
                        break
        return True

    def run(self):
        frames, running = 0, True
        try:
            self.draw()  # Valid hit regions before the first input event.
            while running:
                for event in self.pg.event.get():
                    if not self.event(event):
                        running = False
                        break
                if not running:
                    break
                if not self.paused:
                    self.arena.step(self.speed)
                self.stats_cache = self.arena.stats()
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
