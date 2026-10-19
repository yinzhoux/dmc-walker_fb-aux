"""Live DMC walker: window + FIFO goal / state edits.

On macOS the viewer must be started with mjpython.

  .venv/bin/mjpython play.py --ckpt checkpoints/fb_walker_proto.pt
  echo forward > /tmp/fb_goal
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fb.live import play
from fb.walker import GOAL_POSES


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/fb_walker_proto.pt"))
    parser.add_argument("--pipe", type=Path, default=Path("/tmp/fb_goal"))
    parser.add_argument("--goal", default="stand", choices=list(GOAL_POSES))
    parser.add_argument("--task", default="walker_stand")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    play(_parse())


if __name__ == "__main__":
    main()
