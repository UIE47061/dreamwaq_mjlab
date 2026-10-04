"""Shared directed terrain geometry for mjlab and standalone MuJoCo."""

import math

KINDS = ("flat", "stairs_up", "stairs_down", "slope_up", "slope_down")
PATCH_SIZE = (8.0, 4.0)
START_X, SPAWN_X, CENTER_Y = 2.0, 0.8, 2.0


def layout(kind, difficulty=0.0, height=None, angle=None, depth=0.35):
    height = 0.02 + 0.13 * difficulty if height is None else height
    angle = 25.0 * difficulty if angle is None else angle
    stairs = kind.startswith("stairs")
    end = START_X + (5 * depth if stairs else 3.0)
    rise = 5 * height if stairs else 3 * math.tan(math.radians(angle))
    if kind == "flat":
        rise = 0.0
    down = kind.endswith("down")
    initial, final = (rise, 0.0) if down else (0.0, rise)
    boxes = []

    def slab(left, right, top):
        boxes.append(
            {
                "pos": ((left + right) / 2, CENTER_Y, top / 2 - 0.1),
                "size": ((right - left) / 2, 2.0, top / 2 + 0.1),
                "quat": (1.0, 0.0, 0.0, 0.0),
            }
        )

    slab(0.0, START_X, initial)
    if stairs:
        for i in range(5):
            z = initial - (i + 1) * height if down else (i + 1) * height
            slab(START_X + i * depth, START_X + (i + 1) * depth, max(z, 0.0))
    elif kind == "flat":
        slab(START_X, end, 0.0)
    else:
        theta = math.atan2(final - initial, 3.0)
        boxes.append(
            {
                "pos": (
                    (START_X + end) / 2 + 0.1 * math.sin(theta),
                    CENTER_Y,
                    (initial + final) / 2 - 0.1 * math.cos(theta),
                ),
                "size": (1.5 / math.cos(theta), 2.0, 0.1),
                "quat": (math.cos(theta / 2), 0.0, -math.sin(theta / 2), 0.0),
            }
        )
    slab(end, 8.0, final)
    return boxes, (SPAWN_X, CENTER_Y, initial), end + 0.6 - SPAWN_X


def add_geometries(body, boxes):
    import mujoco

    return [
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            **box,
            friction=(1.0, 0.005, 0.0005),
            group=0,
            contype=1,
            conaffinity=1,
            rgba=(0.35, 0.45, 0.55, 1),
        )
        for box in boxes
    ]
