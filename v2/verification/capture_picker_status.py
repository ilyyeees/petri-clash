"""Capture checkpoint-policy feedback with real bundled models and dummy SDL."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from arena import Arena, ArenaUI, parse_args
from clash import list_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("pinned", "fallback", "research"), default="pinned")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("picker-status.png"))
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    selections = {
        "pinned": ["--left", "1", "--left-seed", "0", "--right", "2", "--right-seed", "1"],
        "fallback": ["--lesson", "--left", "2", "--left-seed", "1", "--right", "7", "--right-seed", "2"],
        "research": ["--allow-unhealthy", "--left", "2", "--left-seed", "0", "--right", "1", "--right-seed", "0"],
    }
    world = Arena(parse_args(["--device", "cpu", "--seed", "42", "--window-size", str(args.width),
                              *selections[args.scenario]]), list_targets())
    ui = ArenaUI(world)
    try:
        if args.scenario == "fallback":
            ui.action("lesson")
            ui.action("right")
            assert world.args.right_seed == 2
            assert world.right["seed_dir"] == "seed_001"
        world.step(160)
        ui.paused = True
        if args.height is not None:
            ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=args.width, h=args.height))
        ui.draw(dt=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(ui.window, str(args.output))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
