"""Adapted source model with explicit motor and joint mapping."""

from dataclasses import dataclass

import mujoco
import torch
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.actuator.pd_actuator import IdealPdActuator, pd_torque
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from dreamwaq_mjlab.control import DEFAULT, JOINTS, KD, KP, ROBOT_XML, TORQUE_LIMIT


@dataclass(kw_only=True)
class StrengthPdCfg(IdealPdActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return StrengthPd(self, entity, target_ids, target_names)


class StrengthPd(IdealPdActuator):
    param_names = ("stiffness", "damping", "force_limit", "strength")

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.strength = torch.ones_like(self.stiffness)

    @staticmethod
    def control_law(params, cmd):
        torque = pd_torque(params["stiffness"], params["damping"], cmd) * params["strength"]
        return torque.clamp(-params["force_limit"], params["force_limit"])


def motor_strength(env, env_ids):
    ids = slice(None) if env_ids is None else env_ids
    # Fused actuator fields are live views into the group's parameters.
    for actuator in env.scene["robot"].actuators:
        if hasattr(actuator, "strength"):
            actuator.strength[ids] = torch.empty_like(actuator.strength[ids]).uniform_(0.9, 1.1)


def get_spec():
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    # Source torque sensors refer to joints, so removing old motors is safe.
    return spec


def get_little_white_v3_robot_cfg():
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0, 0, 0.28), joint_pos=dict(zip(JOINTS, DEFAULT)), joint_vel={".*": 0.0}
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                StrengthPdCfg(
                    target_names_expr=(joint,),
                    stiffness=KP,
                    damping=KD,
                    effort_limit=TORQUE_LIMIT,
                    delay_min_lag=0,
                    delay_max_lag=3,
                    delay_update_period=200,
                )
                for joint in JOINTS
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision",),
                contype=1,
                conaffinity=1,
                condim={".*_Foot_collision": 3, ".*_collision": 1},
                priority={".*_Foot_collision": 1, ".*": 0},
                friction={".*_Foot_collision": (1.0, 0.005, 0.0005)},
                solref=(0.01, 1.0),
            ),
        ),
    )


LITTLE_WHITE_V3_ACTION_SCALE = 0.25
