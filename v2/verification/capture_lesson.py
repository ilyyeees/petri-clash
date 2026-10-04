"""Reproduce one regrowth-lesson stage with shipped weights and offscreen SDL."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygame
from arena import Arena, ArenaUI, parse_args
from clash import list_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("seed", "grow", "damage", "injured", "recover", "complete"), default="injured")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("lesson-injured.png"))
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int)
    parser.add_argument("--culture", type=int, default=1)
    args = parser.parse_args()
    world = Arena(parse_args(["--device", "cpu", "--seed", "42", "--lesson",
                              "--left", str(args.culture), "--window-size", str(args.width)]), list_targets())
    ui = ArenaUI(world)
    try:
        if args.height is not None:
            ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=args.width, h=args.height))
        ui.draw(dt=0)
        if args.phase != "seed":
            ui.action("lesson_primary")
            world.step(80 if args.phase == "grow" else 160)
            ui.draw(dt=.1)
        if args.phase in ("injured", "recover", "complete"):
            ui.action("lesson_primary")
            ui.draw(dt=.1)
        if args.phase in ("recover", "complete"):
            ui.action("lesson_primary")
            world.step(3 if args.phase == "recover" else 160)
            ui.draw(dt=.1)
        ui.paused = True
        ui.draw(dt=.05)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(ui.window, str(args.output))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
