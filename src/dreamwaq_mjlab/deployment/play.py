"""Play a checkpoint through its exported CENet + actor in standalone MuJoCo."""

import argparse
from pathlib import Path

from dreamwaq_mjlab.deployment.export import load_policy
from dreamwaq_mjlab.deployment.sim2sim import run
from dreamwaq_mjlab.terrain_geometry import KINDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--terrain", choices=KINDS, default="stairs_up")
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--angle", type=float, default=25.0)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", type=Path)
    a = p.parse_args()
    run(load_policy(a.checkpoint), a.terrain, a.height, a.angle, a.seconds, a.headless, a.video)


if __name__ == "__main__":
    main()
