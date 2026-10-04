"""Reproduce the feedback screenshot with shipped models and an offscreen UI."""
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
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("arena-feedback.png"))
    args = parser.parse_args()
    world = Arena(parse_args(["--device", "cpu", "--seed", "42",
                              "--left-pos", "18,24", "--right-pos", "30,24"]), list_targets())
    ui = ArenaUI(world)
    try:
        for _ in range(160):
            world.step()
            ui.draw(dt=1 / 30)
        ui.paused = True
        ui.action(("view", "territory"))
        ui.draw(dt=.1)
        ui.interact(23, 30)
        ui.draw(dt=.16)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(ui.window, str(args.output))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
