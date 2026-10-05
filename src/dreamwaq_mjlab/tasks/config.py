"""Multidirectional rough-terrain velocity tracking environment."""

import math
from dataclasses import asdict

from mjlab.envs.mdp import dr
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg

from dreamwaq_mjlab.checkpoint import COMMAND_FRAME, TASK_VERSION
from dreamwaq_mjlab.control import JOINTS, ROBOT_XML
from dreamwaq_mjlab.tasks import mdp
from dreamwaq_mjlab.tasks.base import little_white_v3_rough_env_cfg
from dreamwaq_mjlab.tasks.robot import motor_strength
from dreamwaq_mjlab.tasks.terrains import terrain_cfg


def task_spec(cfg, task, seed, beta):
    return {
        "task_version": TASK_VERSION,
        "command_frame": COMMAND_FRAME,
        "task": task,
        "seed": seed,
        "beta": beta,
        "reward_revision": "anti_bounce_v1",
        "num_envs": cfg.scene.num_envs,
        "terrain": asdict(cfg.scene.terrain.terrain_generator),
        "command": asdict(cfg.commands["twist"]),
    }


def make_cfg(
    num_envs=512, flat=False, randomized=True, kind=None, difficulty=None, seed=42, curriculum=True
):
    c = little_white_v3_rough_env_cfg()
    c.scene.num_envs = num_envs
    c.scene.terrain.terrain_generator = terrain_cfg(flat, kind, difficulty, seed)
    c.scene.terrain.max_init_terrain_level = 5 if curriculum else None
    c.seed = seed
    c.sim.mujoco.timestep = 0.005
    c.decimation = 4
    c.episode_length_s = 20.0
    c.sim.njmax = 300
    c.sim.nconmax = 128
    c.actions["joint_pos"].actuator_names = JOINTS
    c.actions["joint_pos"].preserve_order = True
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    c.actions["joint_pos"].clip = {
        n: tuple(float(x) for x in m.jnt_range[m.joint(n).id]) for n in JOINTS
    }
    # Group-local observation noise; clean values are separate CENet targets.
    c.observations["actor"] = ObservationGroupCfg(
        terms={
            "proprio": ObservationTermCfg(
                func=mdp.proprio, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02)
            )
        },
        enable_corruption=randomized,
    )
    c.observations["clean"] = ObservationGroupCfg(
        terms={"proprio": ObservationTermCfg(func=mdp.proprio)}, enable_corruption=False
    )
    c.observations["velocity"] = ObservationGroupCfg(
        terms={"velocity": ObservationTermCfg(func=mdp.velocity)}, enable_corruption=False
    )
    # Privileged critic retains terrain and contact signals from the base config.
    c.observations["critic"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=JOINTS, preserve_order=True
    )
    c.observations["critic"].enable_corruption = False
    c.curriculum = (
        {
            "terrain": CurriculumTermCfg(
                func=velocity_mdp.terrain_levels_vel, params={"command_name": "twist"}
            )
        }
        if curriculum and difficulty is None
        else {}
    )
    c.terminations["illegal_contact"].params["sensor_name"] = "trunk_ground_touch"
    c.terminations["illegal_contact"].params["force_threshold"] = 1.0
    c.rewards["thigh_collision"] = RewardTermCfg(
        func=velocity_mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": "thigh_ground_touch"},
    )
    # Commands live in the yaw frame, even when the robot tilts on terrain.
    c.rewards["track_linear_velocity"].func = mdp.track_planar_velocity
    c.rewards["track_linear_velocity"].params["std"] = 0.5
    c.rewards["track_angular_velocity"].func = mdp.track_yaw_velocity
    c.rewards["track_angular_velocity"].params["std"] = 0.5
    # Soft anti-bounce costs, without imposing a fixed gait on rough terrain.
    c.rewards["vertical_bounce"] = RewardTermCfg(
        func=mdp.vertical_bounce, weight=-1.0, params={"deadband": 0.2}
    )
    c.rewards["prolonged_flight"] = RewardTermCfg(
        func=mdp.prolonged_flight,
        weight=-1.0,
        params={"sensor_name": "feet_ground_contact", "grace_time": 0.08},
    )
    c.rewards["body_ang_vel"].weight = -0.05
    from mjlab.envs import mdp as env_mdp

    c.terminations["fell_over"] = TerminationTermCfg(
        func=env_mdp.bad_orientation, params={"limit_angle": 1.22}
    )
    c.events["reset_base"].params["pose_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.01, 0.05),
        "yaw": (-math.pi, math.pi),
    }
    cmd = c.commands["twist"]
    cmd.heading_command = True
    cmd.ranges.heading = (-math.pi, math.pi)
    cmd.heading_control_stiffness = 0.5
    cmd.resampling_time_range = (10.0, 10.0)
    cmd.rel_standing_envs = 0.2
    cmd.rel_heading_envs = 1.0
    cmd.rel_world_envs = 0.0
    cmd.rel_forward_envs = 0.0
    cmd.init_velocity_prob = 0.0
    cmd.ranges.lin_vel_x = (-0.6, 1.0)
    cmd.ranges.lin_vel_y = (-0.5, 0.5)
    cmd.ranges.ang_vel_z = (-1.57, 1.57)
    # Joint density + COM perturbation preserves a physically valid inertia tensor.
    c.events.pop("base_com", None)
    c.events["base_inertia"] = EventTermCfg(
        func=dr.pseudo_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            "alpha_range": (-0.0526802578, 0.0476550899),
            "t1_range": (-0.025, 0.025),
            "t2_range": (-0.025, 0.025),
            "t3_range": (-0.03, 0.03),
        },
    )
    c.events["pd_gains"] = EventTermCfg(
        func=dr.pd_gains, mode="startup", params={"kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1)}
    )
    c.events["motor_strength"] = EventTermCfg(func=motor_strength, mode="startup")
    if not randomized:
        c.events = {k: v for k, v in c.events.items() if k in ("reset_base", "reset_robot_joints")}
        for a in c.scene.entities["robot"].articulation.actuators:
            a.delay_max_lag = 0
    return c
