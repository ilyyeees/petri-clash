"""Render real-model lab controls with a deterministic offscreen pointer cue."""

import argparse
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from arena import Arena, ArenaUI, parse_args
from clash import list_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("damage", "plant_left", "plant_right"), default="plant_right")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int)
    parser.add_argument("--reduced-motion", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("lab-tools.png"))
    args = parser.parse_args()
    argv = ["--device", "cpu", "--seed", "42", "--window-size", str(args.width)]
    if args.reduced_motion:
        argv.append("--reduced-motion")
    world = Arena(parse_args(argv), list_targets())
    ui = ArenaUI(world)
    try:
        world.step(160)
        ui.paused = True
        if args.height is not None:
            ui.event(pygame.event.Event(pygame.VIDEORESIZE, w=args.width, h=args.height))
        ui.action(("tool", args.tool))
        ui.draw(dt=0)
        scale = ui.board.width / world.size
        pointer = (round(ui.board.x + 24.5 * scale), round(ui.board.y + 10.5 * scale))
        with patch("pygame.mouse.get_pos", return_value=pointer), \
                patch("pygame.mouse.get_pressed", return_value=(False, False, False)), \
                patch("pygame.key.get_mods", return_value=0):
            ui.draw(dt=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(ui.window, str(args.output))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
