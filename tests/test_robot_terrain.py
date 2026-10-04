import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dreamwaq_mjlab.control import ASSET_DIR, JOINTS, ROBOT_XML, TORQUE_LIMIT
from dreamwaq_mjlab.deployment.sim2sim import Sim2Sim, make_scene
from dreamwaq_mjlab.learning.models import CENet, DreamActor, DreamPolicy, Normalizer
from dreamwaq_mjlab.terrain_geometry import layout


def test_source_geometry_inertial_and_limits_preserved():
    source = ET.parse(ASSET_DIR / "little_white_v3.xml")
    adapted = ET.parse(ROBOT_XML)
    for tag in ("body", "joint", "inertial"):
        a = source.findall(".//" + tag)
        b = adapted.findall(".//" + tag)
        assert [x.attrib for x in a] == [x.attrib for x in b]
    old = mujoco.MjModel.from_xml_path(str(ASSET_DIR / "little_white_v3.xml"))
    new = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    np.testing.assert_allclose(old.body_mass, new.body_mass)
    np.testing.assert_allclose(old.body_inertia, new.body_inertia)
    for joint in JOINTS:
        np.testing.assert_allclose(
            old.jnt_range[old.joint(joint).id], new.jnt_range[new.joint(joint).id]
        )


@pytest.mark.parametrize("kind", ["stairs_up", "stairs_down", "slope_up", "slope_down"])
def test_directed_course_has_correct_start_and_end(kind):
    boxes, origin, goal = layout(kind, height=0.15, angle=25.0)
    start_top = boxes[0]["pos"][2] + boxes[0]["size"][2]
    end_top = boxes[-1]["pos"][2] + boxes[-1]["size"][2]
    assert start_top == pytest.approx(origin[2])
    assert (end_top > start_top) == kind.endswith("up")
    assert goal > 3.0
    model, data, *_ = make_scene(kind)
    assert np.isfinite(data.qpos).all()


def test_sim2sim_mapping_and_torque_limits():
    policy = DreamPolicy(Normalizer(45), CENet(), DreamActor()).eval()
    sim = Sim2Sim(policy)
    assert sim.history.shape == (6, 45)
    assert sim.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert sim.model.opt.ccd_iterations == 50
    assert sim.model.opt.ls_iterations == 20
    for leg in ("FR", "FL", "RL", "RR"):
        assert sim.model.geom(f"{leg}_Foot_collision").priority == 1
    ids = [sim.model.joint(n).id for n in JOINTS]
    assert [ids[i] for i in sim.actuator_mapping] == sim.model.actuator_trnid[:, 0].tolist()
    for _ in range(5):
        sim.advance()
    assert sim.max_torque <= TORQUE_LIMIT
    assert np.isfinite(sim.data.qpos).all()


def test_sim2sim_observation_uses_current_integrated_pose():
    policy = DreamPolicy(Normalizer(45), CENet(), DreamActor()).eval()
    sim = Sim2Sim(policy)
    sim.data.qvel[3:6] = [1.0, 0.5, 0.2]
    sim.advance()
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, sim.data.qpos[3:7])
    expected_gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(sim.history[-1, 3:6], expected_gravity, atol=1e-7)


def test_sim2sim_world_command_is_expressed_in_body_yaw_frame():
    policy = DreamPolicy(Normalizer(45), CENet(), DreamActor()).eval()
    sim = Sim2Sim(policy, command=0.4)
    sim.data.qpos[3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    mujoco.mj_forward(sim.model, sim.data)
    np.testing.assert_allclose(sim.observation()[6:9], [0.0, -0.4, 0.0], atol=1e-7)
