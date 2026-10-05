"""Play rough-task checkpoints in the same mjlab environment used for training."""

import argparse
import json
import os
from pathlib import Path

import torch

from dreamwaq_mjlab.deployment.export import load_policy
from dreamwaq_mjlab.deployment.viewer_runtime import join_native_viewer_threads
from dreamwaq_mjlab.learning.history import History
from dreamwaq_mjlab.tasks.terrains import KINDS


class HistoryPolicy:
    """Viewer adapter: maintain six frames and reset only freshly reset worlds."""

    def __init__(self, env, policy):
        self.env, self.policy = env, policy
        self.history = None

    def reset(self):
        self.history = None

    @torch.no_grad()
    def __call__(self, obs):
        first = obs["actor"]
        if self.history is None:
            self.history = History(first)
        else:
            self.history.append(first)
            ids = (self.env.episode_length_buf == 0).nonzero().flatten()
            self.history.reset(ids, first)
        return self.policy(self.history.data)


class RecordingEnv:
    """Stream frames after each second control step, keeping memory bounded."""

    def __init__(self, env, writer):
        self.env, self.writer = env, writer
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        result = self.env.step(actions)
        if self.steps % 2 == 0:
            self.writer.append_data(self.env.render())
        self.steps += 1
        return result


def run(
    checkpoint,
    terrain="mixed",
    difficulty=None,
    num_envs=16,
    device="cuda:0",
    viewer="auto",
    seconds=60.0,
    headless=False,
    video=None,
    seed=2026,
):
    from mjlab.envs import ManagerBasedRlEnv

    from dreamwaq_mjlab.tasks.config import make_cfg

    if num_envs < 1 or seconds <= 0:
        raise ValueError("num-envs and seconds must be positive")
    policy = load_policy(checkpoint, device)
    cfg = make_cfg(
        num_envs, kind=terrain, difficulty=difficulty, seed=seed, randomized=False, curriculum=False
    )
    cfg.viewer.distance = 5.0
    cfg.viewer.elevation = -30.0
    cfg.viewer.width, cfg.viewer.height = 640, 480
    cfg.commands["twist"].debug_vis = not headless
    if headless:
        for sensor in cfg.scene.sensors:
            sensor.debug_vis = False
    env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array" if video else None)
    writer = None
    try:
        env.reset(seed=seed)
        wrapped = env
        if video:
            import imageio

            Path(video).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video), fps=25)
            wrapped = RecordingEnv(env, writer)
        adapter = HistoryPolicy(env, policy)
        steps = round(seconds / env.step_dt)
        if headless:
            with torch.no_grad():
                for _ in range(steps):
                    actions = adapter(wrapped.get_observations())
                    if not torch.isfinite(actions).all():
                        raise RuntimeError("Nonfinite policy output")
                    wrapped.step(actions)
        else:
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

            backend = (
                ("native" if os.environ.get("DISPLAY") else "viser") if viewer == "auto" else viewer
            )
            cls = NativeMujocoViewer if backend == "native" else ViserPlayViewer
            player = cls(wrapped, adapter)
            try:
                player.run(num_steps=steps)
            finally:
                if backend == "native":
                    join_native_viewer_threads()
            if player._last_error is not None:
                raise RuntimeError(player._last_error)
        result = {
            "terrain": terrain,
            "difficulty": difficulty,
            "num_envs": num_envs,
            "seconds": seconds,
            "finite": bool(torch.isfinite(env.sim.data.qpos).all()),
        }
        print(json.dumps(result), flush=True)
        return result
    finally:
        if writer:
            writer.close()
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--terrain", choices=("mixed", "flat", *KINDS), default="mixed")
    p.add_argument("--difficulty", type=float)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    run(
        a.checkpoint,
        a.terrain,
        a.difficulty,
        a.num_envs,
        a.device,
        a.viewer,
        a.seconds,
        a.headless,
        a.video,
        a.seed,
    )


if __name__ == "__main__":
    main()
