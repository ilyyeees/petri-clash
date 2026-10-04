"""Reproduce a seeded-duel screenshot using shipped models and offscreen SDL.

The output is a raster UI verification image, not a native-display benchmark.
"""
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
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("duel-result.png"))
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int)
    parser.add_argument("--mode", choices=("hard", "soft"), default="hard")
    args = parser.parse_args()
    if args.ticks < 0:
        parser.error("--ticks must be nonnegative")
    world = Arena(parse_args(["--device", "cpu", "--seed", "42", "--duel",
                              "--window-size", str(args.width), "--mode", args.mode]), list_targets())
    ui = ArenaUI(world)
    try:
        if args.height is not None:
            ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=args.width, h=args.height))
        world.step(args.ticks)
        ui.view = "territory" if args.mode == "hard" else "organisms"
        ui.draw(dt=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(ui.window, str(args.output))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
