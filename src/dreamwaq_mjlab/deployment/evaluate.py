"""Batched rough-terrain velocity tracking evaluation with estimated-velocity inference."""

import argparse
import json
import random
from functools import wraps
from pathlib import Path

import numpy as np
import torch

from dreamwaq_mjlab.deployment.export import load_policy
from dreamwaq_mjlab.learning.history import History
from dreamwaq_mjlab.tasks.terrains import KINDS

MOTIONS = {
    "forward": (0.6, 0.0, 0.0),
    "backward": (-0.4, 0.0, 0.0),
    "left": (0.0, 0.4, 0.0),
    "right": (0.0, -0.4, 0.0),
    "turn_left": (0.0, 0.0, 0.8),
    "turn_right": (0.0, 0.0, -0.8),
    "standing": (0.0, 0.0, 0.0),
    "random": None,
}
DIFFICULTIES = (0.0, 0.5, 1.0)


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


def tracking_score(xy_error, yaw_error):
    return 0.5 * (torch.exp(-xy_error / 0.25) + torch.exp(-yaw_error / 0.25))


def set_fixed_commands(env, fixed_ids, targets):
    """Keep fixed cases fixed across command timer resampling; random cases retain heading control."""
    cmd = env.command_manager.get_term("twist")
    cmd.is_heading_env[fixed_ids] = False
    cmd.is_standing_env[fixed_ids] = False
    cmd.is_world_env[fixed_ids] = False
    cmd.is_forward_env[fixed_ids] = False
    cmd.vel_command_b[fixed_ids] = targets


def policy_observations(env):
    """Expose interventions immediately without advancing cached sensor/history buffers.

    mjlab caches observations until the next step. Preserve the existing actor
    corruption but replace the command value that was cached before intervention.
    """
    obs = env.get_observations()
    actor = obs["actor"].clone()
    actor[:, 6:9] += env.command_manager.get_command("twist") - obs["clean"][:, 6:9]
    return {**obs, "actor": actor}


@preserve_random_state
def evaluate_policy(
    policy,
    trials=100,
    randomized=False,
    device="cuda:0",
    conditions=None,
    seed=2026,
    terrains=None,
    difficulties=DIFFICULTIES,
    motions=None,
    max_envs=128,
):
    from mjlab.envs import ManagerBasedRlEnv

    from dreamwaq_mjlab.tasks.config import make_cfg
    from dreamwaq_mjlab.tasks.mdp import tracking_errors

    terrains = list(KINDS) if terrains is None else list(terrains)
    motions = list(MOTIONS) if motions is None else list(motions)
    if trials < 1 or max_envs < 1 or not terrains or not difficulties or not motions:
        raise ValueError("Evaluation requires positive trials/max-envs and nonempty conditions")
    if any(k not in (*KINDS, "flat") for k in terrains) or any(m not in MOTIONS for m in motions):
        raise ValueError("Unknown terrain or motion")
    if any(not 0 <= d <= 1 for d in difficulties):
        raise ValueError("Difficulty must be in [0, 1]")
    conditions = (
        [(k, d, m) for k in terrains for d in difficulties for m in motions]
        if (conditions is None)
        else list(conditions)
    )
    if not conditions:
        raise ValueError("No evaluation conditions")
    # One generated map is reused across condition batches, avoiding hundreds of compilations.
    levels = sorted(set(d for _, d, _ in conditions))
    if len(levels) > 2 and levels != [0.0, 0.5, 1.0]:
        raise ValueError("Use a single difficulty or the default 0/0.5/1 grid")
    n = min(max_envs, trials)
    # Keep each batch within a terrain/difficulty row; >= trials can pack multiple motions.
    cases_per_batch = max(1, max_envs // min(trials, max_envs))
    n = min(max_envs, trials * cases_per_batch)
    cfg = make_cfg(n, flat=terrains == ["flat"], randomized=randomized, seed=seed, curriculum=False)
    generator = cfg.scene.terrain.terrain_generator
    if terrains != ["flat"]:
        generator.sub_terrains = {k: generator.sub_terrains[k] for k in terrains}
    generator.num_rows = len(levels)
    generator.num_cols = len(generator.sub_terrains)
    generator.difficulty_range = (min(levels), max(levels))
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.auto_reset = False
    cfg.commands["twist"].debug_vis = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = ManagerBasedRlEnv(cfg, device=device)
    names = list(generator.sub_terrains)
    results = []
    horizon = round(cfg.episode_length_s / env.step_dt)
    try:
        for start in range(0, len(conditions), cases_per_batch):
            batch = conditions[start : start + cases_per_batch]
            # Trials larger than the GPU batch limit are accumulated in chunks.
            records = [[] for _ in batch]
            for trial_start in range(0, trials, max_envs):
                count = min(trials - trial_start, max_envs)
                assignments = [(i, case) for i, case in enumerate(batch) for _ in range(count)]
                assignments = assignments[:n]
                terrain = env.scene.terrain
                for j, (_, (kind, difficulty, _)) in enumerate(assignments):
                    terrain.terrain_types[j] = names.index(kind)
                    terrain.terrain_levels[j] = levels.index(difficulty)
                env.scene.env_origins[:] = terrain.terrain_origins[
                    terrain.terrain_levels, terrain.terrain_types
                ]
                obs, _ = env.reset()
                fixed = [
                    (j, MOTIONS[case[2]])
                    for j, (_, case) in enumerate(assignments)
                    if MOTIONS[case[2]] is not None
                ]
                ids = torch.tensor([j for j, _ in fixed], device=device, dtype=torch.long)
                targets = torch.tensor([v for _, v in fixed], device=device).reshape(-1, 3)
                set_fixed_commands(env, ids, targets)
                obs = policy_observations(env)
                history = History(obs["actor"])
                active = torch.arange(n, device=device) < len(assignments)
                steps = torch.zeros(n, device=device)
                xy_sum, yaw_sum, score_sum = (torch.zeros(n, device=device) for _ in range(3))
                fallen = torch.zeros(n, device=device, dtype=torch.bool)
                boundary = torch.zeros_like(fallen)
                with torch.no_grad():
                    for _ in range(horizon):
                        command = env.command_manager.get_command("twist").clone()
                        actions = policy(history.data)
                        terminal, _, term, trunc, _ = env.step(actions)
                        xy, yaw = tracking_errors(env, command)
                        xy_sum += xy * active
                        yaw_sum += yaw * active
                        score_sum += tracking_score(xy, yaw) * active
                        steps += active
                        fallen |= active & term
                        boundary |= active & env.termination_manager.get_term(
                            "out_of_terrain_bounds"
                        )
                        done = term | trunc
                        active &= ~done
                        if not active.any():
                            break
                        history.append(terminal["actor"])
                        if done.any():
                            reset_ids = done.nonzero().flatten()
                            obs, _ = env.reset(env_ids=reset_ids)
                        set_fixed_commands(env, ids, targets)
                        obs = policy_observations(env)
                        # Command changes are part of the same current frame, not a new history frame.
                        history.data[:, -1] = obs["actor"]
                        if done.any():
                            history.reset(reset_ids, obs["actor"])
                for j, (case_id, _) in enumerate(assignments):
                    records[case_id].append(
                        {
                            "planar_velocity_rmse": float(
                                (xy_sum[j] / steps[j].clamp_min(1)).sqrt()
                            ),
                            "yaw_velocity_rmse": float((yaw_sum[j] / steps[j].clamp_min(1)).sqrt()),
                            "tracking_score": float(score_sum[j] / horizon),
                            "duration_s": float(steps[j] * env.step_dt),
                            "fallen": bool(fallen[j]),
                            "boundary_exit": bool(boundary[j]),
                            "planar_squared_error_sum": float(xy_sum[j]),
                            "yaw_squared_error_sum": float(yaw_sum[j]),
                            "steps": int(steps[j]),
                        }
                    )
            for (kind, difficulty, motion), trials_data in zip(batch, records, strict=True):
                samples = sum(t["steps"] for t in trials_data)
                result = {
                    "terrain": kind,
                    "difficulty": difficulty,
                    "motion": motion,
                    "randomized": randomized,
                    "seed": seed,
                    "trials": len(trials_data),
                    "planar_velocity_rmse": float(
                        np.sqrt(
                            sum(t["planar_squared_error_sum"] for t in trials_data)
                            / max(samples, 1)
                        )
                    ),
                    "yaw_velocity_rmse": float(
                        np.sqrt(
                            sum(t["yaw_squared_error_sum"] for t in trials_data) / max(samples, 1)
                        )
                    ),
                    "tracking_score": float(np.mean([t["tracking_score"] for t in trials_data])),
                    "fallen_rate": float(np.mean([t["fallen"] for t in trials_data])),
                    "boundary_exit_rate": float(np.mean([t["boundary_exit"] for t in trials_data])),
                    "mean_duration_s": float(np.mean([t["duration_s"] for t in trials_data])),
                    "trial_results": trials_data,
                }
                results.append(result)
            print(f"EVAL {len(results)}/{len(conditions)} conditions", flush=True)
    finally:
        env.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--max-envs", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--mode", choices=("nominal", "randomized", "both"), default="both")
    p.add_argument("--terrain", choices=("mixed", "flat", *KINDS), default="mixed")
    p.add_argument("--difficulty", type=float, nargs="+", default=list(DIFFICULTIES))
    p.add_argument("--motion", choices=tuple(MOTIONS), nargs="+", default=list(MOTIONS))
    a = p.parse_args()
    torch.set_num_threads(4)
    policy = load_policy(a.checkpoint, a.device)
    result = []
    for randomized in [False, True] if a.mode == "both" else [a.mode == "randomized"]:
        result.extend(
            evaluate_policy(
                policy,
                a.trials,
                randomized,
                a.device,
                seed=a.seed,
                terrains=None if a.terrain == "mixed" else [a.terrain],
                difficulties=a.difficulty,
                motions=a.motion,
                max_envs=a.max_envs,
            )
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"checkpoint": str(a.checkpoint), "results": result}, indent=2))
    print(
        json.dumps(
            {
                "conditions": len(result),
                "tracking_score": float(np.mean([r["tracking_score"] for r in result])),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
