"""Export the deterministic CENet + actor, with frozen normalization."""

import argparse
import json
from pathlib import Path

import torch

from dreamwaq_mjlab.control import (
    ACTION_SCALE,
    DECIMATION,
    DEFAULT,
    JOINTS,
    KD,
    KP,
    OBS_ORDER,
    PHYSICS_DT,
    SOURCE_COMMIT,
    TORQUE_LIMIT,
)
from dreamwaq_mjlab.learning.models import CENet, DreamActor, DreamPolicy, Normalizer


def load_policy(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    norm = Normalizer(45).to(device)
    norm.load_state_dict(state["normalizer"])
    cenet = CENet().to(device)
    cenet.load_state_dict(state["cenet"])
    actor = DreamActor().to(device)
    actor.load_state_dict(state["actor"])
    return DreamPolicy(norm, cenet, actor).to(device).eval()


def export(checkpoint, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = load_policy(checkpoint)
    scripted = torch.jit.script(policy)
    scripted.save(str(output / "policy.pt"))
    restored = torch.jit.load(str(output / "policy.pt")).eval()
    with torch.no_grad():
        for n in (1, 4):
            h = torch.randn(n, 6, 45)
            torch.testing.assert_close(restored(h), policy(h))
    metadata = {
        "version": 1,
        "checkpoint": str(checkpoint),
        "input_shape": ["batch", 6, 45],
        "output_shape": ["batch", 12],
        "observation_order": OBS_ORDER,
        "command_frame": "world_course_velocity_expressed_in_body_yaw_frame",
        "history_order": "oldest_to_newest",
        "history_reset": "repeat_first_observation",
        "normalization": "embedded",
        "joint_order": JOINTS,
        "default_angles": DEFAULT,
        "kp": KP,
        "kd": KD,
        "torque_limit": TORQUE_LIMIT,
        "action_scale": ACTION_SCALE,
        "physics_dt": PHYSICS_DT,
        "control_decimation": DECIMATION,
        "control_hz": 1.0 / (PHYSICS_DT * DECIMATION),
        "physics_hz": 1.0 / PHYSICS_DT,
        "model_source_url": "https://github.com/morrisx28/csl_mujoco/tree/"
        + SOURCE_COMMIT
        + "/unitree_robots/little_white_v3",
        "model_source_commit": SOURCE_COMMIT,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Exported and verified {output / 'policy.pt'}", flush=True)
    return output / "policy.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    export(a.checkpoint, a.output)


if __name__ == "__main__":
    main()
