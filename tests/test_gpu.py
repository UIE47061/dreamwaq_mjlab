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
