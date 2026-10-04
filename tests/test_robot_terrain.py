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
    ids = [sim.model.joint(n).id for n in JOINTS]
    assert [ids[i] for i in sim.actuator_mapping] == sim.model.actuator_trnid[:, 0].tolist()
    for _ in range(5):
        sim.advance()
    assert sim.max_torque <= TORQUE_LIMIT
    assert np.isfinite(sim.data.qpos).all()
