import torch

from dreamwaq_mjlab.control import DEFAULT, JOINTS


def joint_ids(env):
    if not hasattr(env, "_dwq_joint_ids"):
        env._dwq_joint_ids = [env.scene["robot"].joint_names.index(n) for n in JOINTS]
    return env._dwq_joint_ids


def proprio(env):
    r = env.scene["robot"].data
    ids = joint_ids(env)
    command = env.command_manager.get_command("twist")
    default = torch.tensor(DEFAULT, device=env.device)
    # Encoder bias is a persistent per-episode/environment model of calibration.
    pos = r.joint_pos[:, ids] - default
    pos = pos + r.encoder_bias[:, ids]
    return torch.cat(
        (
            r.root_link_ang_vel_b,
            r.projected_gravity_b,
            command,
            pos,
            r.joint_vel[:, ids],
            env.action_manager.action,
        ),
        dim=-1,
    )


def velocity(env):
    return env.scene["robot"].data.root_link_lin_vel_b


def forward_progress(env):
    command = env.command_manager.get_command("twist")
    return env.scene["robot"].data.root_link_lin_vel_w[:, 0].clamp(-0.5, 1.0) * (
        command[:, 0] > 0.1
    )


def corridor_exit(env):
    pos = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    return (pos[:, 0] > 6.4) | (pos[:, 0] < -0.6) | (pos[:, 1].abs() > 1.8)


def crossed_course(env):
    t = env.scene.terrain
    names = list(t.cfg.terrain_generator.sub_terrains)
    threshold = torch.tensor(
        [3.85 if n.startswith("stairs") else 4.8 for n in names], device=env.device
    )
    x = env.scene["robot"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return x >= threshold[t.terrain_types]


def terrain_curriculum(env, env_ids):
    t = env.scene.terrain
    if env.common_step_counter == 0 or not hasattr(env, "reset_terminated"):
        return {"mean": t.terrain_levels.float().mean()}
    passed = crossed_course(env)[env_ids] & (~env.reset_terminated[env_ids])
    # Failed/blocked episodes demote; simply staying alive is not progression.
    failed = ~passed
    t.update_env_origins(env_ids, passed, failed)
    out = {"mean": t.terrain_levels.float().mean()}
    for i, name in enumerate(t.cfg.terrain_generator.sub_terrains):
        mask = t.terrain_types == i
        out[name] = (
            t.terrain_levels[mask].float().mean()
            if mask.any()
            else torch.tensor(0.0, device=env.device)
        )
    return out
