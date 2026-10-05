"""Render a standalone surface-height preview of all nine native MuJoCo terrains."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from dreamwaq_mjlab.tasks.terrains import KINDS, terrain_cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--output", type=Path, default=Path("artifacts/rough_terrain_preview.png"))
    a = p.parse_args()
    cfg = terrain_cfg(difficulty=a.difficulty)
    fig = plt.figure(figsize=(14, 12), layout="constrained")
    grid = np.linspace(0.05, 7.95, 65)
    x, y = np.meshgrid(grid, grid)
    for index, kind in enumerate(KINDS):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        output = cfg.sub_terrains[kind].function(a.difficulty, spec, np.random.default_rng(42))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        heights = np.empty_like(x)
        geom = np.array([-1], dtype=np.int32)
        for j in np.ndindex(x.shape):
            distance = mujoco.mj_ray(
                model,
                data,
                np.array([x[j], y[j], 10.0]),
                np.array([0.0, 0.0, -1.0]),
                None,
                1,
                -1,
                geom,
            )
            heights[j] = 10.0 - distance if distance >= 0 else np.nan
        ax = fig.add_subplot(3, 3, index + 1, projection="3d")
        ax.plot_surface(x, y, heights, cmap="terrain", linewidth=0, antialiased=False)
        ax.scatter(
            output.origin[0],
            output.origin[1],
            output.origin[2] + 0.03,
            color="red",
            s=35,
            depthshade=False,
            label="spawn",
        )
        ax.set_title(kind)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_zlim(-1.5, 1.5)
        ax.view_init(elev=30, azim=-55)
    fig.suptitle(f"DreamWaQ rough terrain / difficulty={a.difficulty:g} / red=spawn")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=150)
    plt.close(fig)
    print(a.output.resolve())


if __name__ == "__main__":
    main()
