"""Check the shipping regrowth lesson on each ready culture and three seeds.

This is a bounded CPU smoke study, not a ranking or a shape-fidelity test.
The report contains portable culture identities and counts, never host paths.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from arena import Arena, parse_args
from clash import list_targets, target_status
from runtime import runtime_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    rows = []
    targets = list_targets()
    ready = [(index, target) for index, target in enumerate(targets, 1)
             if target_status(target)["status"] == "ready"]
    for index, target in ready:
        for seed in (0, 7, 42):
            world = Arena(parse_args(["--device", "cpu", "--cpu-threads", "1", "--lesson",
                                      "--left", str(index), "--seed", str(seed)]), targets)
            world.plant_lesson()
            assert world.step(1000) == 160 and world.lesson.phase == "damage"
            world.cut_lesson()
            assert world.lesson.phase == "injured"
            rng = torch.get_rng_state().clone()
            assert world.step(1000) == 0 and torch.equal(rng, torch.get_rng_state())
            world.watch_lesson()
            world.step(1000)
            result = world.lesson.snapshot()
            assert result["phase"] == "complete" and result["hold"] == 24
            assert world.stats()["finite"] and not bool(world.b.any())
            frozen = world.a.clone()
            assert world.step(1000) == 0 and torch.equal(frozen, world.a)
            row = {"culture": target.stem, "checkpoint_seed": world.left["seed_dir"],
                   "seed": seed, "grid_size": world.size, "plan": result["plan"],
                   **{key: result[key] for key in ("baseline", "removed", "remaining_after_cut",
                       "target", "recovery_ticks", "current", "hold", "valid")}}
            rows.append(row)
            print(f"{target.stem} / seed {seed}: complete at recovery tick {result['recovery_ticks']}",
                  file=sys.stderr, flush=True)
    if not rows:
        raise SystemExit("No ready cultures found")
    report = {"schema_version": 1, "runtime": runtime_info(), "configuration": result["config"],
              "scope": "Each ready shipped culture, seeds 0/7/42, fixed 48x48 solo lesson. "
                       "Living-cell counts only; no general recovery or shape-fidelity guarantee.",
              "all_complete": True, "cases": rows}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
