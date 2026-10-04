"""Independent Flat and MixedTerrain environment factories."""

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
from mjlab.utils.noise import UniformNoiseCfg

from dreamwaq_mjlab.control import JOINTS, ROBOT_XML
from dreamwaq_mjlab.tasks import mdp
from dreamwaq_mjlab.tasks.base import little_white_v3_rough_env_cfg
from dreamwaq_mjlab.tasks.robot import motor_strength
from dreamwaq_mjlab.tasks.terrains import terrain_cfg


def make_cfg(num_envs=512, flat=False, randomized=True, kind=None, height=None, angle=None):
    c = little_white_v3_rough_env_cfg()
    c.scene.num_envs = num_envs
    c.scene.terrain.terrain_generator = terrain_cfg(flat, kind, height, angle)
    c.scene.terrain.max_init_terrain_level = 0
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
        {"terrain": CurriculumTermCfg(func=mdp.terrain_curriculum)} if kind is None else {}
    )
    c.rewards["progress"] = RewardTermCfg(func=mdp.forward_progress, weight=0.5)
    c.terminations.pop("out_of_terrain_bounds", None)
    c.terminations["corridor_exit"] = TerminationTermCfg(func=mdp.corridor_exit, time_out=False)
    c.terminations["course_complete"] = TerminationTermCfg(func=mdp.crossed_course, time_out=True)
    from mjlab.envs import mdp as env_mdp

    c.terminations["fell_over"] = TerminationTermCfg(
        func=env_mdp.bad_orientation, params={"limit_angle": 1.22}
    )
    c.events["reset_base"].params["pose_range"] = {
        "x": (-0.1, 0.1),
        "y": (-0.15, 0.15),
        "z": (0.01, 0.05),
        "yaw": (-0.15, 0.15),
    }
    cmd = c.commands["twist"]
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_standing_envs = 0.05
    cmd.rel_heading_envs = 0.0
    cmd.rel_world_envs = 1.0
    cmd.rel_forward_envs = 0.0
    cmd.ranges.lin_vel_x = (0.2, 0.6)
    cmd.ranges.lin_vel_y = (-0.05, 0.05)
    cmd.ranges.ang_vel_z = (-0.1, 0.1)
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
        c.events["reset_base"].params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.03, 0.03),
            "yaw": (0.0, 0.0),
        }
        for a in c.scene.entities["robot"].articulation.actuators:
            a.delay_max_lag = 0
    if kind is not None:
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.4, 0.4)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
    return c
