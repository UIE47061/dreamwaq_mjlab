"""CPU MuJoCo deployment. Run with mjpython on macOS for the native viewer."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from dreamwaq_mjlab.control import (
    ACTION_SCALE,
    DECIMATION,
    DEFAULT,
    JOINTS,
    KD,
    KP,
    PHYSICS_DT,
    ROBOT_XML,
    TORQUE_LIMIT,
)
from dreamwaq_mjlab.terrain_geometry import KINDS, add_geometries, layout


def make_scene(kind="flat", height=0.15, angle=25.0, render=False):
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    boxes, origin, goal = layout(kind, height=height, angle=angle)
    add_geometries(spec.worldbody, boxes)
    spec.worldbody.add_light(
        pos=(1.0, 2.0, 5.0), dir=(0.0, 0.0, -1.0), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.worldbody.add_camera(
        name="overview", pos=(-1.0, -4.0, 3.0), xyaxes=(1.0, 0.0, 0.0, 0.0, 0.4, 1.0)
    )
    spec.option.timestep = PHYSICS_DT
    # Nominal training defaults for the common collision contact model.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.iterations = 10
    model = spec.compile()
    data = mujoco.MjData(model)
    ids = [model.joint(n).id for n in JOINTS]
    qadr = np.array([model.jnt_qposadr[i] for i in ids])
    vadr = np.array([model.jnt_dofadr[i] for i in ids])
    data.qpos[:3] = np.array(origin) + np.array([0.0, 0.0, 0.31])
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = DEFAULT
    mujoco.mj_forward(model, data)
    return model, data, qadr, vadr, origin, goal


class Sim2Sim:
    def __init__(self, policy, kind="flat", height=0.15, angle=25.0, command=0.4):
        self.policy = policy.cpu().eval()
        self.command = np.array([command, 0.0, 0.0], dtype=np.float32)
        self.model, self.data, self.qadr, self.vadr, self.origin, self.goal = make_scene(
            kind, height, angle
        )
        self.base = self.model.body("base_link").id
        ids = [self.model.joint(n).id for n in JOINTS]
        self.ranges = self.model.jnt_range[ids]
        self.actuator_mapping = np.array(
            [ids.index(int(j)) for j in self.model.actuator_trnid[:, 0]]
        )
        self.action = np.zeros(12, dtype=np.float32)
        self.history = np.repeat(self.observation()[None], 6, axis=0)
        self.max_torque = 0.0
        self.max_progress = 0.0

    def observation(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        omega = rotation.T @ self.data.sensor("frame_ang_vel").data
        # No frame_vel: velocity is inferred by CENet, including in sim2sim.
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        obs = np.concatenate(
            (
                omega,
                gravity,
                self.command,
                self.data.qpos[self.qadr] - DEFAULT,
                self.data.qvel[self.vadr],
                self.action,
            )
        ).astype(np.float32)
        if obs.shape != (45,) or not np.isfinite(obs).all():
            raise RuntimeError("Invalid proprioception")
        return obs

    def advance(self):
        with torch.no_grad():
            self.action = self.policy(torch.from_numpy(self.history[None])).squeeze(0).numpy()
        if not np.isfinite(self.action).all():
            raise RuntimeError("Nonfinite action")
        target = np.clip(
            np.array(DEFAULT) + ACTION_SCALE * self.action, self.ranges[:, 0], self.ranges[:, 1]
        )
        for _ in range(DECIMATION):
            torque = np.clip(
                KP * (target - self.data.qpos[self.qadr]) - KD * self.data.qvel[self.vadr],
                -TORQUE_LIMIT,
                TORQUE_LIMIT,
            )
            self.data.ctrl[:] = torque[self.actuator_mapping]
            mujoco.mj_step(self.model, self.data)
            self.max_torque = max(self.max_torque, float(np.abs(torque).max()))
        self.history = np.concatenate((self.history[1:], self.observation()[None]), axis=0)
        self.max_progress = max(self.max_progress, float(self.data.qpos[0] - self.origin[0]))


def run(policy, kind="flat", height=0.15, angle=25.0, seconds=20.0, headless=False, video=None):
    sim = Sim2Sim(policy, kind, height, angle)
    viewer = None
    renderer = None
    frames = []
    if not headless:
        from mujoco import viewer as viewer_module

        viewer = viewer_module.launch_passive(sim.model, sim.data)
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
    if video:
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
    for i in range(int(seconds / 0.02)):
        start = time.monotonic()
        sim.advance()
        if renderer is not None and i % 2 == 0:
            renderer.update_scene(sim.data, camera="overview")
            frames.append(renderer.render().copy())
        if viewer:
            if not viewer.is_running():
                break
            viewer.cam.lookat[:] = sim.data.qpos[:3]
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.monotonic() - start)))
    if viewer:
        viewer.close()
    if renderer:
        import imageio.v3 as iio

        Path(video).parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(video, np.stack(frames), fps=25)
        renderer.close()
    metrics = {
        "kind": kind,
        "height_cm": height * 100,
        "angle_deg": angle,
        "max_forward_m": sim.max_progress,
        "goal_forward_m": sim.goal,
        "max_torque_nm": sim.max_torque,
        "final_height_m": float(sim.data.qpos[2]),
        "finite": bool(np.isfinite(sim.data.qpos).all()),
    }
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", type=Path)
    p.add_argument("--terrain", choices=KINDS, default="flat")
    p.add_argument("--height", type=float, default=0.15)
    p.add_argument("--angle", type=float, default=25.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    policy = torch.jit.load(str(a.policy), map_location="cpu").eval()
    run(policy, a.terrain, a.height, a.angle, a.seconds, a.headless, a.video)


if __name__ == "__main__":
    main()
