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


def vertical_bounce(env, deadband=0.2):
    """Allow gentle elevation changes; penalize rapid world-frame bouncing."""
    vz = env.scene["robot"].data.root_link_lin_vel_w[:, 2]
    return (vz.abs() - deadband).clamp_min(0).square()


def prolonged_flight(env, sensor_name, grace_time=0.08):
    """Penalize all-feet flight after a short terrain-transition allowance."""
    data = env.scene[sensor_name].data
    assert data.current_air_time is not None
    assert data.found is not None
    all_airborne = (data.found == 0).reshape(env.num_envs, -1).all(dim=-1)
    flight_time = data.current_air_time.reshape(env.num_envs, -1).amin(dim=-1)
    return all_airborne.float() * (flight_time > grace_time).float()


def tracking_errors(env, command=None):
    """Planar yaw-frame and world yaw-rate squared errors."""
    r = env.scene["robot"].data
    command = env.command_manager.get_command("twist") if command is None else command
    yaw = r.heading_w
    vx, vy = r.root_link_lin_vel_w[:, 0], r.root_link_lin_vel_w[:, 1]
    actual = torch.stack(
        (yaw.cos() * vx + yaw.sin() * vy, -yaw.sin() * vx + yaw.cos() * vy), dim=-1
    )
    xy = (actual - command[:, :2]).square().sum(-1)
    angular = (r.root_link_ang_vel_w[:, 2] - command[:, 2]).square()
    return xy, angular


def track_planar_velocity(env, std, command_name):
    xy, _ = tracking_errors(env, env.command_manager.get_command(command_name))
    return torch.exp(-xy / std**2)


def track_yaw_velocity(env, std, command_name):
    _, angular = tracking_errors(env, env.command_manager.get_command(command_name))
    return torch.exp(-angular / std**2)
