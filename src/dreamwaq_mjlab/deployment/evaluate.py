"""Directed course completion tests; privileged inputs never reach the policy."""

import argparse
import json
import random
from functools import wraps
from pathlib import Path

import numpy as np
import torch

from dreamwaq_mjlab.deployment.export import load_policy
from dreamwaq_mjlab.learning.history import History
from dreamwaq_mjlab.terrain_geometry import layout

CONDITIONS = [
    (kind, height, None)
    for kind in ("stairs_up", "stairs_down")
    for height in (0.02, 0.06, 0.10, 0.15)
] + [(kind, None, angle) for kind in ("slope_up", "slope_down") for angle in (10.0, 20.0, 25.0)]


def preserve_random_state(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                return function(*args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    return wrapped


@preserve_random_state
def evaluate_policy(
    policy, trials=100, randomized=False, device="cuda:0", conditions=None, seed=2026
):
    from mjlab.envs import ManagerBasedRlEnv

    from dreamwaq_mjlab.tasks.config import make_cfg

    results = []
    for condition in CONDITIONS if conditions is None else conditions:
        kind, height, angle = condition
        c = make_cfg(trials, randomized=randomized, kind=kind, height=height, angle=angle)
        # Nominal trials vary only starting pose; no model/sensor/push perturbation.
        c.events["reset_base"].params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.15, 0.15),
            "z": (0.01, 0.05),
            "yaw": (-0.15, 0.15),
        }
        c.auto_reset = False
        c.seed = seed
        c.scene.terrain.terrain_generator.seed = seed
        e = ManagerBasedRlEnv(c, device=device)
        obs, _ = e.reset(seed=seed)
        history = History(obs["actor"])
        n = e.num_envs
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        success = torch.zeros_like(finished)
        fallen = torch.zeros_like(finished)
        stalled = torch.zeros_like(finished)
        reasons = {
            name: torch.zeros_like(finished)
            for name in ("illegal_contact", "fell_over", "corridor_exit")
        }
        durations = torch.zeros(n, device=device)
        maxx = torch.zeros_like(durations)
        anchor = e.scene["robot"].data.root_link_pos_w[:, 0].clone()
        anchor_steps = torch.zeros(n, dtype=torch.long, device=device)
        _, _, goal = layout(kind, height=height, angle=angle)
        with torch.no_grad():
            for step in range(1000):
                action = policy(history.data)
                action[finished] = 0.0
                terminal, _reward, term, trunc, _ = e.step(action)
                pos = e.scene["robot"].data.root_link_pos_w
                dx = pos[:, 0] - e.scene.env_origins[:, 0]
                maxx = torch.where(~finished, torch.maximum(maxx, dx), maxx)
                reached = (dx >= goal) & (~term) & (~finished)
                anchor_steps += 1
                blocked = (anchor_steps >= 150) & (pos[:, 0] - anchor < 0.1) & (~finished)
                refresh = anchor_steps >= 150
                anchor[refresh] = pos[refresh, 0]
                anchor_steps[refresh] = 0
                blocked = blocked & (~reached)
                failed = term & (~finished)
                newly = (reached | failed | blocked | trunc) & (~finished)
                success |= reached
                fallen |= failed
                for name, mask in reasons.items():
                    mask |= e.termination_manager.get_term(name) & (~finished)
                stalled |= blocked
                durations[newly] = (step + 1) * 0.02
                finished |= newly
                history.append(terminal["actor"])
                obs = terminal
                if finished.all():
                    break
                ids = (term | trunc | blocked | reached).nonzero().flatten()
                if len(ids):
                    obs, _ = e.reset(env_ids=ids)
                    history.reset(ids, obs["actor"])
                    anchor[ids] = e.scene["robot"].data.root_link_pos_w[ids, 0]
                    anchor_steps[ids] = 0
        durations[~finished] = 20.0
        result = {
            "terrain": kind,
            "height_cm": None if height is None else height * 100,
            "angle_deg": angle,
            "randomized": randomized,
            "seed": seed,
            "trials": n,
            "success_rate": int(success.sum()) / n,
            "successes": int(success.sum()),
            "fallen_rate": int(fallen.sum()) / n,
            "stalled_rate": int(stalled.sum()) / n,
            "mean_max_forward_m": float(maxx.mean()),
            "mean_duration_s": float(durations.mean()),
            "goal_forward_m": goal,
            "failure_reasons": {name: int(mask.sum()) / n for name, mask in reasons.items()},
            "trial_results": [
                {
                    "success": bool(success[j]),
                    "failed": bool(fallen[j]),
                    "stalled": bool(stalled[j]),
                    "max_forward_m": float(maxx[j]),
                    "duration_s": float(durations[j]),
                }
                for j in range(n)
            ],
        }
        results.append(result)
        print("EVAL " + json.dumps(result), flush=True)
        e.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--mode", choices=("nominal", "randomized", "both"), default="both")
    a = p.parse_args()
    torch.set_num_threads(4)
    policy = load_policy(a.checkpoint, a.device)
    result = []
    for randomized in [False, True] if a.mode == "both" else [a.mode == "randomized"]:
        result.extend(evaluate_policy(policy, a.trials, randomized, a.device, seed=a.seed))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"checkpoint": str(a.checkpoint), "results": result}, indent=2))


if __name__ == "__main__":
    main()
