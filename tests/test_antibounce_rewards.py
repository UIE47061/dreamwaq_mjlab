from types import SimpleNamespace

import torch

from dreamwaq_mjlab.tasks.mdp import prolonged_flight, vertical_bounce


def test_vertical_cost_allows_slow_height_changes_and_is_symmetric():
    data = SimpleNamespace(
        root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 0.1], [0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    torch.testing.assert_close(vertical_bounce(env), torch.tensor([0.0, 0.09, 0.09]))


def test_flight_cost_allows_brief_flight_and_any_supporting_foot():
    data = SimpleNamespace(
        found=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
        current_air_time=torch.tensor([[0.04] * 4, [0.12] * 4, [0.0, 0.2, 0.2, 0.2]]),
    )
    env = SimpleNamespace(num_envs=3, scene={"feet": SimpleNamespace(data=data)})
    torch.testing.assert_close(prolonged_flight(env, "feet"), torch.tensor([0.0, 1.0, 0.0]))
