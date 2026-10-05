"""CPU MuJoCo deployment. Run with mjpython on macOS for the native viewer."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from dreamwaq_mjlab.checkpoint import COMMAND_FRAME, TASK_VERSION
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
from dreamwaq_mjlab.deployment.viewer_runtime import join_native_viewer_threads


def make_scene():
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    origin = (0.0, 0.0, 0.0)
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(100.0, 100.0, 0.1),
        friction=(1.0, 0.005, 0.0005),
        rgba=(0.35, 0.45, 0.55, 1),
    )
    spec.worldbody.add_light(
        pos=(1.0, 2.0, 5.0), dir=(0.0, 0.0, -1.0), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.worldbody.add_camera(
        name="overview", pos=(-1.0, -4.0, 3.0), xyaxes=(1.0, 0.0, 0.0, 0.0, 0.4, 1.0)
    )
    spec.option.timestep = PHYSICS_DT
    # Nominal training defaults for the common collision contact model.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.impratio = 10.0
    spec.option.iterations = 10
    spec.option.tolerance = 1e-8
    spec.option.ls_iterations = 20
    spec.option.ls_tolerance = 0.01
    spec.option.ccd_iterations = 50
    for geom in spec.geoms:
        if geom.name.endswith("_Foot_collision"):
            geom.priority = 1
    model = spec.compile()
    data = mujoco.MjData(model)
    ids = [model.joint(n).id for n in JOINTS]
    qadr = np.array([model.jnt_qposadr[i] for i in ids])
    vadr = np.array([model.jnt_dofadr[i] for i in ids])
    data.qpos[:3] = np.array(origin) + np.array([0.0, 0.0, 0.31])
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = DEFAULT
    mujoco.mj_forward(model, data)
    return model, data, qadr, vadr, origin


class Sim2Sim:
    def __init__(self, policy, command=(0.4, 0.0, 0.0)):
        self.policy = policy.cpu().eval()
        self.command = np.array(command, dtype=np.float32)
        if self.command.shape != (3,):
            raise ValueError("command must contain vx, vy, yaw-rate")
        self.model, self.data, self.qadr, self.vadr, self.origin = make_scene()
        self.base = self.model.body("base_link").id
        ids = [self.model.joint(n).id for n in JOINTS]
        self.ranges = self.model.jnt_range[ids]
        self.actuator_mapping = np.array(
            [ids.index(int(j)) for j in self.model.actuator_trnid[:, 0]]
        )
        self.action = np.zeros(12, dtype=np.float32)
        self.history = np.repeat(self.observation()[None], 6, axis=0)
        self.max_torque = 0.0
        self.illegal_contact = False

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
        self.illegal_contact = False
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
            # Match the trunk ground-contact termination across physics substeps.
            for index, contact in enumerate(self.data.contact):
                geoms = (int(contact.geom1), int(contact.geom2))
                terrain_side = any(self.model.geom_bodyid[g] == 0 for g in geoms)
                trunk_side = any(self.model.geom(g).name == "base_link_collision" for g in geoms)
                if terrain_side and trunk_side:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, force)
                    self.illegal_contact |= bool(np.linalg.norm(force[:3]) > 1.0)
            self.max_torque = max(self.max_torque, float(np.abs(torque).max()))
        # Refresh poses and sensors after integration, as mjlab does before observations.
        mujoco.mj_forward(self.model, self.data)
        self.history = np.concatenate((self.history[1:], self.observation()[None]), axis=0)


def run(policy, seconds=20.0, headless=False, video=None, command=(0.4, 0.0, 0.0)):
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    sim = Sim2Sim(policy, command=command)
    viewer = None
    renderer = None
    writer = None
    status = "completed"
    elapsed = 0.0
    camera = mujoco.MjvCamera()
    camera.distance = 4.0
    camera.azimuth = 90.0
    camera.elevation = -25.0
    try:
        if not headless:
            from mujoco import viewer as viewer_module

            viewer = viewer_module.launch_passive(sim.model, sim.data)
            viewer.cam.distance = 4.0
            viewer.cam.azimuth, viewer.cam.elevation = 90, -25
        if video:
            import imageio

            Path(video).parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(sim.model, height=480, width=640)
            writer = imageio.get_writer(str(video), fps=25)
        for i in range(round(seconds / 0.02)):
            start = time.monotonic()
            sim.advance()
            if renderer is not None and i % 2 == 0:
                camera.lookat[:] = sim.data.qpos[:3]
                renderer.update_scene(sim.data, camera=camera)
                writer.append_data(renderer.render())
            if viewer:
                if not viewer.is_running():
                    break
                viewer.cam.lookat[:] = sim.data.qpos[:3]
                viewer.sync()
                time.sleep(max(0.0, 0.02 - (time.monotonic() - start)))
            elapsed = (i + 1) * 0.02
            upright = sim.data.xmat[sim.base].reshape(3, 3)[2, 2]
            if (
                not np.isfinite(sim.data.qpos).all()
                or upright < np.cos(1.22)
                or sim.illegal_contact
            ):
                status = "fallen"
                break
    finally:
        if viewer:
            viewer.close()
            join_native_viewer_threads()
        if writer:
            writer.close()
        if renderer:
            renderer.close()
    metrics = {
        "terrain": "flat",
        "command": list(command),
        "command_frame": COMMAND_FRAME,
        "max_torque_nm": sim.max_torque,
        "final_height_m": float(sim.data.qpos[2]),
        "finite": bool(np.isfinite(sim.data.qpos).all()),
        "status": status,
        "duration_s": elapsed,
    }
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", type=Path)
    p.add_argument(
        "--command", type=float, nargs=3, default=(0.4, 0.0, 0.0), metavar=("VX", "VY", "YAW_RATE")
    )
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", type=Path)
    a = p.parse_args()
    metadata = json.loads(a.policy.with_name("metadata.json").read_text())
    if (
        metadata.get("task_version") != TASK_VERSION
        or metadata.get("command_frame") != COMMAND_FRAME
    ):
        raise ValueError("Export a new rough-task checkpoint before running sim2sim")
    torch.set_num_threads(2)
    policy = torch.jit.load(str(a.policy), map_location="cpu").eval()
    run(policy, a.seconds, a.headless, a.video, a.command)


if __name__ == "__main__":
    main()
