import copy
import math

import pytest
import torch

from dreamwaq_mjlab.learning.history import History, adaboot_probability, estimator_loss
from dreamwaq_mjlab.learning.models import CENet, DreamActor, DreamPolicy, Normalizer, mlp
from dreamwaq_mjlab.learning.ppo import returns_advantages, update


def test_history_local_reset_and_order():
    first = torch.arange(90, dtype=torch.float32).reshape(2, 45)
    h = History(first)
    h.append(first + 100)
    assert torch.equal(h.data[:, 0], first)
    assert torch.equal(h.data[:, -1], first + 100)
    before = h.data[1].clone()
    h.reset(torch.tensor([0]), first - 100)
    assert torch.equal(h.data[0], (first[0] - 100).repeat(6, 1))
    assert torch.equal(h.data[1], before)


@pytest.mark.parametrize(
    "values,expected",
    [([], 0.0), ([1.0], 0.0), ([0.0, 0.0], 0.0), ([-1.0, -2.0], 0.0), ([3.0, 3.0], 1.0)],
)
def test_adaboot_edges(values, expected):
    assert adaboot_probability(values) == expected


def test_adaboot_variation_lowers_bootstrap():
    assert 0 <= adaboot_probability([1.0, 10.0]) < adaboot_probability([5.0, 6.0]) <= 1


def test_terminal_target_is_masked():
    net = CENet()
    hist = torch.randn(4, 6, 45)
    velocity = torch.randn(4, 3)
    target = torch.randn(4, 45)
    valid = torch.tensor([True, False, True, False])
    torch.manual_seed(1)
    loss1, parts1 = estimator_loss(net, hist, velocity, target, valid)
    target[~valid] = 1e6
    torch.manual_seed(1)
    loss2, parts2 = estimator_loss(net, hist, velocity, target, valid)
    torch.testing.assert_close(loss1, loss2)
    loss2.backward()
    assert sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None) > 0
    torch.manual_seed(1)
    _, parts = estimator_loss(net, hist, velocity, target, torch.zeros_like(valid))
    assert parts[1] == 0


def test_gae_terminal_and_timeout_bootstrap():
    # Values of terminal next states are masked by the caller; timeouts retain them.
    r = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    v = torch.zeros_like(r)
    n = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
    done = torch.ones_like(r, dtype=torch.bool)
    returns, adv = returns_advantages(r, v, n, done)
    torch.testing.assert_close(returns, torch.tensor([[1.0, 2.98], [1.0, 1.0]]))


def test_normalizer_population_statistics():
    n = Normalizer(3)
    x = torch.randn(100, 3)
    n.update(x[:30])
    n.update(x[30:])
    torch.testing.assert_close(n.mean, x.mean(0))
    torch.testing.assert_close(n.var, x.var(0, unbiased=False))


def test_export_is_pure_proprioception_and_matches(tmp_path):
    norm = Normalizer(45)
    norm.update(torch.randn(100, 45))
    net = CENet()
    actor = DreamActor()
    policy = DreamPolicy(norm, net, actor).eval()
    script = torch.jit.script(policy)
    script.save(str(tmp_path / "policy.pt"))
    loaded = torch.jit.load(str(tmp_path / "policy.pt"))
    for batch in (1, 4):
        h = torch.randn(batch, 6, 45)
        assert loaded(h).shape == (batch, 12)
        torch.testing.assert_close(policy(h), loaded(h))
    with pytest.raises((torch.jit.Error, RuntimeError)):
        loaded(torch.zeros(1, 6, 48))


def test_ppo_and_estimator_both_update_and_optimizers_resume(tmp_path):
    torch.set_num_threads(2)
    actor = DreamActor()
    critic = mlp([10, 512, 256, 128, 1])
    net = CENet()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    ceopt = torch.optim.Adam(net.parameters(), lr=1e-3)
    features = torch.randn(2, 8, 64)
    critic_input = torch.randn(2, 8, 10)
    with torch.no_grad():
        dist = actor.distribution(features)
        actions = dist.sample()
        data = {
            "features": features,
            "critic": critic_input,
            "actions": actions,
            "means": dist.loc,
            "stds": dist.scale.expand_as(dist.loc),
            "log_probs": dist.log_prob(actions).sum(-1),
            "values": critic(critic_input).squeeze(-1),
            "next_values": torch.zeros(2, 8),
            "rewards": torch.randn(2, 8),
            "dones": torch.zeros(2, 8, dtype=torch.bool),
            "history": torch.randn(2, 8, 6, 45),
            "velocity": torch.randn(2, 8, 3),
            "next_clean": torch.randn(2, 8, 45),
        }
    before = copy.deepcopy(actor.state_dict())
    cebefore = copy.deepcopy(net.state_dict())
    lr, losses = update(actor, critic, net, opt, ceopt, data, 1e-3, epochs=1, minibatches=2)
    assert all(math.isfinite(x) for x in losses.values())
    assert any(not torch.equal(before[k], v) for k, v in actor.state_dict().items())
    assert any(not torch.equal(cebefore[k], v) for k, v in net.state_dict().items())
    saved = {"optimizer": opt.state_dict(), "ce_optimizer": ceopt.state_dict()}
    torch.save(saved, tmp_path / "checkpoint.pt")
    restored = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    opt.load_state_dict(restored["optimizer"])
    ceopt.load_state_dict(restored["ce_optimizer"])
    assert len(opt.state) > 0 and len(ceopt.state) > 0


def test_evaluation_does_not_change_training_rng():
    import random

    import numpy as np

    from dreamwaq_mjlab.deployment.evaluate import preserve_random_state

    @preserve_random_state
    def evaluation():
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        random.random()
        np.random.rand()
        torch.rand(10)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    evaluation()
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert expected[0:2] == actual[0:2]
    torch.testing.assert_close(expected[2], actual[2])
