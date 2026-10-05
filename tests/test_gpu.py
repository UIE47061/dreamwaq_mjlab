"""Small end-to-end GPU check, explicit opt-in via pytest -m gpu."""

import pytest
import torch

from dreamwaq_mjlab.control import JOINTS


@pytest.mark.gpu
def test_environment_action_order_randomization_and_manual_reset():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from mjlab.envs import ManagerBasedRlEnv

    from dreamwaq_mjlab.tasks.config import make_cfg

    c = make_cfg(8)
    c.auto_reset = False
    e = ManagerBasedRlEnv(c, device="cuda:0")
    try:
        obs, _ = e.reset()
        action = e.action_manager.get_term("joint_pos")
        assert tuple(action._target_names) == JOINTS
        assert obs["actor"].shape == (8, 45) and obs["velocity"].shape == (8, 3)
        strengths = torch.cat([a.strength for a in e.scene["robot"].actuators], -1)
        assert (strengths >= 0.9).all() and (strengths <= 1.1).all()
        assert strengths.std() > 0
        # True velocity is absent from the explicit actor observation function.
        from dreamwaq_mjlab.tasks import mdp

        before = mdp.proprio(e).clone()
        assert torch.equal(before[:, 6:9], e.command_manager.get_command("twist"))
        assert torch.equal(before[:, 33:], e.action_manager.action)
        for _ in range(3):
            obs, rew, term, trunc, _ = e.step(torch.zeros(8, 12, device="cuda:0"))
            assert torch.isfinite(rew).all()
            ids = (term | trunc).nonzero().flatten()
            if len(ids):
                e.reset(env_ids=ids)
        e.reset(env_ids=torch.tensor([0], device="cuda:0"))
    finally:
        e.close()


@pytest.mark.gpu
def test_rough_map_scans_commands_and_terminations():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from mjlab.envs import ManagerBasedRlEnv

    from dreamwaq_mjlab.deployment.evaluate import policy_observations, set_fixed_commands
    from dreamwaq_mjlab.tasks.config import make_cfg

    cfg = make_cfg(9, randomized=False, curriculum=False, difficulty=0.5)
    cfg.auto_reset = False
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0, 0),
        "y": (0, 0),
        "z": (0.03, 0.03),
        "yaw": (0, 0),
    }
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        terrain = env.scene.terrain
        terrain.terrain_types[:] = torch.arange(9, device=env.device)
        terrain.terrain_levels[:] = 0
        env.scene.env_origins[:] = terrain.terrain_origins[0, terrain.terrain_types]
        obs, _ = env.reset()
        scan = env.scene["terrain_scan"].data
        assert torch.isfinite(obs["critic"]).all()
        assert torch.isfinite(scan.hit_pos_w).all() and (scan.distances >= 0).all()
        # Raycast and native geometry agree at the center of each patch.
        offset = scan.hit_pos_w[:, :, :2] - env.scene.env_origins[:, None, :2]
        nearest = offset.square().sum(-1).argmin(-1)
        center_z = scan.hit_pos_w[torch.arange(9, device=env.device), nearest, 2]
        assert (center_z - env.scene.env_origins[:, 2]).abs().max() < 0.15
        ids = torch.arange(9, device=env.device)
        commands = torch.tensor(
            [
                [-0.4, 0, 0],
                [0, 0.4, 0],
                [0, -0.4, 0],
                [0, 0, 0.8],
                [0, 0, -0.8],
                [0, 0, 0],
                [0.4, 0, 0],
                [-0.2, 0.2, 0.4],
                [0.2, -0.2, -0.4],
            ],
            device=env.device,
        )
        set_fixed_commands(env, ids, commands)
        obs = policy_observations(env)
        torch.testing.assert_close(obs["actor"][:, 6:9], commands)
        # Changing world yaw must leave direct body commands unchanged.
        env.command_manager.get_term("twist")._update_command()
        torch.testing.assert_close(env.command_manager.get_command("twist"), commands)
        _, _, term, trunc, _ = env.step(torch.zeros(9, 12, device=env.device))
        assert not term.any() and not trunc.any()
        # Truncation occurs only outside the entire grid plus its 20 m border.
        pose = env.scene["robot"].data.root_link_pose_w.clone()
        pose[:, 0] = 1000
        env.scene["robot"].write_root_link_pose_to_sim(pose)
        env.sim.forward()
        from mjlab.tasks.velocity.mdp import out_of_terrain_bounds

        assert out_of_terrain_bounds(env).all()
    finally:
        env.close()
