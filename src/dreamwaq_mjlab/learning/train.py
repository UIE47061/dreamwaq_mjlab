"""Single-GPU DreamWaQ runner with explicit pre-reset terminal observations."""

import argparse
import json
import random
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dreamwaq_mjlab.checkpoint import COMMAND_FRAME, TASK_VERSION, validate_checkpoint
from dreamwaq_mjlab.control import SOURCE_COMMIT
from dreamwaq_mjlab.learning.history import History, adaboot_probability
from dreamwaq_mjlab.learning.models import CENet, DreamActor, DreamPolicy, Normalizer, mlp
from dreamwaq_mjlab.learning.ppo import update


def load_models(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    validate_checkpoint(state)
    normalizer = Normalizer(45).to(device)
    normalizer.load_state_dict(state["normalizer"])
    cenet = CENet().to(device)
    cenet.load_state_dict(state["cenet"])
    actor = DreamActor().to(device)
    actor.load_state_dict(state["actor"])
    return state, normalizer, cenet, actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument(
        "--iterations", type=int, default=10000, help="Additional iterations for this invocation"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=("flat", "mixed"), default="mixed")
    parser.add_argument("--run-name", default="little_white_v3")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.eval_interval) < 1:
        parser.error("num-envs, iterations and eval-interval must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    from mjlab.envs import ManagerBasedRlEnv

    from dreamwaq_mjlab.tasks.config import make_cfg, task_spec
    from dreamwaq_mjlab.tasks.mdp import tracking_errors

    cfg = make_cfg(args.num_envs, flat=args.task == "flat", seed=args.seed)
    cfg.seed = args.seed
    cfg.auto_reset = False
    spec = task_spec(cfg, args.task, args.seed, args.beta)
    if args.resume:
        validate_checkpoint(torch.load(args.resume, map_location="cpu", weights_only=False), spec)
    try:
        env = ManagerBasedRlEnv(cfg, device=args.device)
    except (torch.OutOfMemoryError, MemoryError):
        if args.num_envs != 512:
            raise
        torch.cuda.empty_cache()
        args.num_envs = 256
        cfg = make_cfg(256, flat=args.task == "flat", seed=args.seed)
        cfg.seed = args.seed
        cfg.auto_reset = False
        env = ManagerBasedRlEnv(cfg, device=args.device)
    spec = task_spec(cfg, args.task, args.seed, args.beta)
    now = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d_%H-%M-%S")
    logdir = Path("logs") / f"{now}_{args.run_name}"
    logdir.mkdir(parents=True)
    writer = SummaryWriter(str(logdir))

    def serial(obj):
        if isinstance(obj, Path):
            return str(obj)
        if callable(obj):
            return f"{obj.__module__}.{obj.__qualname__}"
        if isinstance(obj, slice):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    (logdir / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "task_spec": spec,
                "env": asdict(cfg),
                "model_source": SOURCE_COMMIT,
                "mjlab_commit": "b517e0c489139e7fcee95702cfb2b01931264985",
            },
            default=serial,
            indent=2,
        )
    )
    normalizer = Normalizer(45).to(args.device)
    cenet = CENet().to(args.device)
    actor = DreamActor().to(args.device)
    obs, _ = env.reset()
    critic_dim = obs["critic"].shape[1]
    critic_norm = Normalizer(critic_dim).to(args.device)
    critic = mlp([critic_dim, 512, 256, 128, 1]).to(args.device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    ce_optimizer = torch.optim.Adam(cenet.parameters(), lr=1e-3)
    lr = 1e-3
    start_it = 0
    episodes = deque(maxlen=512)
    if args.resume:
        state, normalizer, cenet, actor = load_models(args.resume, args.device)
        critic.load_state_dict(state["critic"])
        critic_norm.load_state_dict(state["critic_normalizer"])
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
        ce_optimizer = torch.optim.Adam(cenet.parameters(), lr=1e-3)
        optimizer.load_state_dict(state["optimizer"])
        ce_optimizer.load_state_dict(state["ce_optimizer"])
        lr = state["learning_rate"]
        start_it = state["iteration"] + 1
        episodes.extend(state["episode_returns"])
        env.common_step_counter = state["env_step"]
        for key in ("terrain_levels", "terrain_types"):
            value = state[key].to(args.device)
            if len(value) != env.num_envs:
                raise ValueError("Resume requires same num-envs")
            getattr(env.scene.terrain, key).copy_(value)
        env.scene.env_origins[:] = env.scene.terrain.terrain_origins[
            env.scene.terrain.terrain_levels, env.scene.terrain.terrain_types
        ]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        # A reconstructed episode must not promote/demote restored terrain levels.
        saved_step = env.common_step_counter
        env.common_step_counter = 0
        obs, _ = env.reset()
        env.common_step_counter = saved_step
    normalizer.update(obs["actor"])
    critic_norm.update(obs["critic"])
    history = History(obs["actor"])
    episode_return = torch.zeros(env.num_envs, device=args.device)
    episode_steps = torch.zeros_like(episode_return)
    recent_lengths = deque(maxlen=512)
    recent_falls = deque(maxlen=512)
    best_score = state.get("best_score", -1.0) if args.resume else -1.0
    begin = time.monotonic()

    def save(iteration, name):
        state = {
            "format_version": 2,
            "task_version": TASK_VERSION,
            "command_frame": COMMAND_FRAME,
            "task_spec": spec,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "cenet": cenet.state_dict(),
            "normalizer": normalizer.state_dict(),
            "critic_normalizer": critic_norm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ce_optimizer": ce_optimizer.state_dict(),
            "iteration": iteration,
            "learning_rate": lr,
            "episode_returns": list(episodes),
            "env_step": env.common_step_counter,
            "terrain_levels": env.scene.terrain.terrain_levels,
            "terrain_types": env.scene.terrain.terrain_types,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "config": vars(args),
            "critic_dim": critic_dim,
            "best_score": best_score,
            "probe_version": 3,
        }
        tmp = logdir / (name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(logdir / name)

    print(f"RUN_DIR={logdir.resolve()}", flush=True)
    try:
        for it in range(start_it, start_it + args.iterations):
            boot = adaboot_probability(list(episodes))
            rollout = {}
            raw_obs = []
            raw_critic = []
            saturated = []
            planar_errors, angular_errors = [], []
            for _ in range(24):
                with torch.no_grad():
                    h = normalizer(history.data)
                    v, mu, logvar = cenet.encode(h)
                    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                    choose = torch.rand(env.num_envs, 1, device=args.device) < boot
                    used_velocity = torch.where(choose, v, obs["velocity"])
                    features = torch.cat((h[:, -1], used_velocity, z), -1)
                    cr = critic_norm(obs["critic"])
                    values = critic(cr).squeeze(-1)
                    dist = actor.distribution(features)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions).sum(-1)
                    velocity = obs["velocity"].clone()
                    command = env.command_manager.get_command("twist").clone()
                    terminal, rewards, term, trunc, _extras = env.step(actions)
                    done = term | trunc
                    # Terminal observation is read before any reset, including timeout bootstrap.
                    next_values = critic(critic_norm(terminal["critic"])).squeeze(-1) * (~term)
                    next_clean = normalizer(terminal["clean"])
                    episode_steps += 1
                    xy_error, yaw_error = tracking_errors(env, command)
                    planar_errors.append(xy_error)
                    angular_errors.append(yaw_error)
                    episode_return += rewards
                    for i in done.nonzero().flatten().tolist():
                        episodes.append(float(episode_return[i]))
                        recent_lengths.append(float(episode_steps[i]))
                        recent_falls.append(float(term[i]))
                    values_to_store = {
                        "history": h,
                        "features": features,
                        "critic": cr,
                        "actions": actions,
                        "log_probs": log_probs,
                        "means": dist.loc,
                        "stds": dist.scale.expand_as(dist.loc),
                        "values": values,
                        "next_values": next_values,
                        "rewards": rewards,
                        "dones": done,
                        "velocity": velocity,
                        "next_clean": next_clean,
                    }
                    for key, value in values_to_store.items():
                        rollout.setdefault(key, []).append(value.detach().clone())
                    raw_obs.append(obs["actor"].clone())
                    raw_critic.append(obs["critic"].clone())
                    torque = env.scene["robot"].data.actuator_force
                    saturated.append((torque.abs() > 26.9).float().mean())
                    history.append(terminal["actor"])
                    obs = terminal
                    if done.any():
                        ids = done.nonzero().flatten()
                        obs, _ = env.reset(env_ids=ids)
                        history.reset(ids, obs["actor"])
                        episode_return[ids] = 0.0
                        episode_steps[ids] = 0.0
            data = {k: torch.stack(v) for k, v in rollout.items()}
            lr, losses = update(actor, critic, cenet, optimizer, ce_optimizer, data, lr, args.beta)
            normalizer.update(torch.stack(raw_obs))
            critic_norm.update(torch.stack(raw_critic))
            stats = {
                **losses,
                "adaboot": boot,
                "episode_reward": float(np.mean(episodes)) if episodes else 0.0,
                "episode_seconds": float(np.mean(recent_lengths)) * 0.02 if recent_lengths else 0.0,
                "planar_velocity_rmse": float(torch.stack(planar_errors).mean().sqrt()),
                "yaw_velocity_rmse": float(torch.stack(angular_errors).mean().sqrt()),
                "failure_fraction": float(np.mean(recent_falls)) if recent_falls else 0.0,
                "torque_saturation": float(torch.stack(saturated).mean()),
                "learning_rate": lr,
                "terrain_level": float(env.scene.terrain.terrain_levels.float().mean()),
            }
            for key, value in stats.items():
                writer.add_scalar(key, value, it)
            for j, name in enumerate(env.scene.terrain.cfg.terrain_generator.sub_terrains):
                mask = env.scene.terrain.terrain_types == j
                if mask.any():
                    writer.add_scalar(
                        "terrain/" + name, env.scene.terrain.terrain_levels[mask].float().mean(), it
                    )
            if it % 50 == 0 or it == start_it + args.iterations - 1:
                with (logdir / "metrics.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            {"iteration": it, "elapsed_seconds": time.monotonic() - begin, **stats}
                        )
                        + "\n"
                    )
                print(json.dumps({"iteration": it, **stats}), flush=True)
                save(it, "latest.pt")
            if (it + 1) % args.eval_interval == 0:
                # Evaluate estimated-velocity inference, not privileged training actions.
                from dreamwaq_mjlab.deployment.evaluate import evaluate_policy

                policy = DreamPolicy(normalizer, cenet, actor).to(args.device).eval()
                probe = evaluate_policy(
                    policy,
                    trials=5,
                    randomized=False,
                    device=args.device,
                    terrains=["flat"] if args.task == "flat" else None,
                )
                score = float(np.mean([x["tracking_score"] for x in probe]))
                writer.add_scalar("evaluation/tracking_score", score, it)
                (logdir / f"probe_{it + 1}.json").write_text(json.dumps(probe, indent=2))
                if score > best_score:
                    best_score = score
                    save(it, "best.pt")
                save(it, f"model_{it + 1}.pt")
        save(it, "final.pt")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
