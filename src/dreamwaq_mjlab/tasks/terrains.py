from dataclasses import dataclass

import numpy as np
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

from dreamwaq_mjlab.terrain_geometry import KINDS, PATCH_SIZE, add_geometries, layout


@dataclass(kw_only=True)
class DirectedTerrainCfg(SubTerrainCfg):
    kind: str = "flat"
    height: float | None = None
    angle: float | None = None
    random_depth: bool = True

    def function(self, difficulty, spec, rng):
        depth = float(rng.uniform(0.30, 0.40)) if self.random_depth else 0.35
        boxes, origin, _ = layout(self.kind, difficulty, self.height, self.angle, depth)
        return TerrainOutput(
            origin=np.asarray(origin),
            geometries=[
                TerrainGeometry(geom=g) for g in add_geometries(spec.body("terrain"), boxes)
            ],
        )


def terrain_cfg(flat=False, kind=None, height=None, angle=None):
    kinds = ("flat",) if flat else ((kind,) if kind else KINDS)
    return TerrainGeneratorCfg(
        seed=42,
        size=PATCH_SIZE,
        num_rows=1 if kind else 10,
        num_cols=len(kinds),
        border_width=1.0,
        curriculum=True,
        add_lights=True,
        sub_terrains={
            k: DirectedTerrainCfg(
                kind=k,
                proportion=1 / len(kinds),
                height=height,
                angle=angle,
                random_depth=kind is None,
            )
            for k in kinds
        },
    )
