# DoublePitTerrainCfg adapts Isaac Lab 2.1 pit_terrain.
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Eight-meter rough terrain patches, adapted from Quadruped_Dwaq_Lab."""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomGridTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

KINDS = (
    "stairs_up_28",
    "stairs_up_32",
    "stairs_down_30",
    "stairs_down_34",
    "boxes",
    "random_rough",
    "wave",
    "slope",
    "high_platform",
)
PATCH_SIZE = (8.0, 8.0)


@dataclass(kw_only=True)
class DoublePitTerrainCfg(SubTerrainCfg):
    """Isaac Lab 2.1 pit_terrain(double_pit=True), using solid boxes.

    The historical name high_platform actually denotes a pit: center -2d,
    inner ring -d, outer ring 0. A solid floor handles difficulty zero.
    """

    pit_depth_range: tuple[float, float] = (0.0, 0.3)
    platform_width: float = 2.0

    def function(self, difficulty, spec, rng):
        del rng
        depth = self.pit_depth_range[0] + difficulty * (
            self.pit_depth_range[1] - self.pit_depth_range[0]
        )
        body = spec.body("terrain")
        geometries = []

        def box(left, right, bottom, top, z_top):
            floor = -2 * depth - 1.0
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=((left + right) / 2, (bottom + top) / 2, (floor + z_top) / 2),
                size=((right - left) / 2, (top - bottom) / 2, (z_top - floor) / 2),
                friction=(1.0, 0.005, 0.0005),
            )
            geometries.append(TerrainGeometry(geom=geom, color=(0.35, 0.45, 0.55, 1)))

        def ring(outer, inner, z_top):
            cx, cy = self.size[0] / 2, self.size[1] / 2
            ox, oy = outer[0] / 2, outer[1] / 2
            ix, iy = inner[0] / 2, inner[1] / 2
            box(cx - ox, cx + ox, cy + iy, cy + oy, z_top)
            box(cx - ox, cx + ox, cy - oy, cy - iy, z_top)
            box(cx - ox, cx - ix, cy - iy, cy + iy, z_top)
            box(cx + ix, cx + ox, cy - iy, cy + iy, z_top)

        middle = tuple(self.platform_width + (s - self.platform_width) * 0.6 for s in self.size)
        box(0, self.size[0], 0, self.size[1], -2 * depth)
        if depth > 0:
            ring(self.size, middle, 0.0)
            ring(middle, (self.platform_width, self.platform_width), -depth)
        return TerrainOutput(
            origin=np.array([self.size[0] / 2, self.size[1] / 2, -2 * depth]),
            geometries=geometries,
        )


def rough_terrains():
    stairs = dict(
        step_height_range=(0.0, 0.23),
        platform_width=3.0,
        border_width=1.0,
        holes=False,
        proportion=0.1,
    )
    hf = dict(horizontal_scale=0.1, vertical_scale=0.005)
    return {
        "stairs_up_28": BoxPyramidStairsTerrainCfg(step_width=0.28, **stairs),
        "stairs_up_32": BoxPyramidStairsTerrainCfg(step_width=0.32, **stairs),
        "stairs_down_30": BoxInvertedPyramidStairsTerrainCfg(step_width=0.30, **stairs),
        "stairs_down_34": BoxInvertedPyramidStairsTerrainCfg(step_width=0.34, **stairs),
        "boxes": BoxRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25, **hf
        ),
        "wave": HfWaveTerrainCfg(proportion=0.1, amplitude_range=(0.0, 0.2), num_waves=5, **hf),
        "slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, inverted=False, **hf
        ),
        "high_platform": DoublePitTerrainCfg(proportion=0.15),
    }


def terrain_cfg(flat=False, kind=None, difficulty=None, seed=42):
    if difficulty is not None and not 0 <= difficulty <= 1:
        raise ValueError("difficulty must be in [0, 1]")
    terrains = {"flat": BoxFlatTerrainCfg()} if flat or kind == "flat" else rough_terrains()
    if kind is not None and kind != "mixed":
        if kind not in terrains:
            raise ValueError(f"Unknown terrain: {kind}")
        terrains = {kind: terrains[kind]}
    for terrain in terrains.values():
        terrain.size = PATCH_SIZE
    return TerrainGeneratorCfg(
        seed=seed,
        size=PATCH_SIZE,
        num_rows=10 if difficulty is None else 1,
        num_cols=len(terrains),
        border_width=20.0,
        border_height=1.0,
        curriculum=True,
        add_lights=True,
        sub_terrains=terrains,
        difficulty_range=(0.0, 1.0) if difficulty is None else (difficulty, difficulty),
    )
