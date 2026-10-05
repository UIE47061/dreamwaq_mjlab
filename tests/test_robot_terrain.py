import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from dreamwaq_mjlab.control import ASSET_DIR, JOINTS, ROBOT_XML, TORQUE_LIMIT
from dreamwaq_mjlab.deployment.sim2sim import Sim2Sim
from dreamwaq_mjlab.learning.models import CENet, DreamActor, DreamPolicy, Normalizer
from dreamwaq_mjlab.tasks.terrains import KINDS, terrain_cfg


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
    for field in ("geom_type", "geom_size", "geom_pos", "geom_quat"):
        np.testing.assert_allclose(getattr(old, field), getattr(new, field))
    for joint in JOINTS:
        np.testing.assert_allclose(
            old.jnt_range[old.joint(joint).id], new.jnt_range[new.joint(joint).id]
        )


def compile_patch(kind, difficulty):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = terrain_cfg(kind=kind, difficulty=difficulty).sub_terrains[kind]
    output = cfg.function(difficulty, spec, np.random.default_rng(42))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, output


def surface_height(model, data, x, y):
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([x, y, 10.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geom
    )
    assert distance >= 0
    return 10.0 - distance


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_rough_terrain_compiles_and_origin_matches_surface(kind, difficulty):
    model, data, output = compile_patch(kind, difficulty)
    assert np.isfinite(model.geom_size).all() and (model.geom_size > 0).all()
    x, y, z = output.origin
    assert x == pytest.approx(4) and y == pytest.approx(4)
    # Heightfield origin is a conservative neighborhood maximum for safe spawn.
    center = surface_height(model, data, x, y)
    assert abs(center - z) < (0.25 if kind == "wave" else 0.06)


@pytest.mark.parametrize("kind", KINDS[:4])
def test_pyramid_has_symmetric_stairs_in_four_directions(kind):
    model, data, output = compile_patch(kind, 1.0)
    for offset in (0.6, 1.8, 2.6, 3.4):
        heights = [
            surface_height(model, data, 4 + offset, 4),
            surface_height(model, data, 4 - offset, 4),
            surface_height(model, data, 4, 4 + offset),
            surface_height(model, data, 4, 4 - offset),
        ]
        np.testing.assert_allclose(heights, heights[0], atol=1e-6)
    outside = surface_height(model, data, 7.4, 4)
    assert (output.origin[2] > outside) == kind.startswith("stairs_up")


def test_double_pit_matches_isaaclab_two_levels():
    model, data, output = compile_patch("high_platform", 1.0)
    assert output.origin[2] == pytest.approx(-0.6)
    assert surface_height(model, data, 4, 4) == pytest.approx(-0.6)
    assert surface_height(model, data, 5.5, 4) == pytest.approx(-0.3)
    assert surface_height(model, data, 7.5, 4) == pytest.approx(0.0)
    model, data, _ = compile_patch("high_platform", 0.0)
    assert surface_height(model, data, 5.5, 4) == pytest.approx(0.0)


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


def test_sim2sim_body_command_is_independent_of_world_heading():
    policy = DreamPolicy(Normalizer(45), CENet(), DreamActor()).eval()
    sim = Sim2Sim(policy, command=(0.4, -0.2, 0.8))
    sim.data.qpos[3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    mujoco.mj_forward(sim.model, sim.data)
    np.testing.assert_allclose(sim.observation()[6:9], [0.4, -0.2, 0.8], atol=1e-7)
