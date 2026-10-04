"""CENet, asymmetric PPO models, and deterministic deployment graph."""

import copy

import torch
from torch import nn


def mlp(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        linear = nn.Linear(a, b)
        nn.init.orthogonal_(linear.weight, gain=0.01 if i == len(sizes) - 2 else 1.414)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    @torch.jit.ignore
    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.numel())
        count = x.shape[0]
        mean = x.mean(0)
        var = x.var(0, unbiased=False)
        delta = mean - self.mean
        total = self.count + count
        self.var.copy_(
            (self.var * self.count + var * count + delta.square() * self.count * count / total)
            / total
        )
        self.mean.add_(delta * (count / total))
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var.clamp_min(1e-4))).clamp(-10.0, 10.0)


class CENet(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = mlp([270, 128, 64])
        self.velocity = nn.Linear(64, 3)
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = mlp([latent_dim + 3, 64, 128, 45])

    def encode(self, history):
        h = self.encoder(history.flatten(start_dim=1))
        return self.velocity(h), self.mu(h), self.logvar(h).clamp(-10.0, 5.0)

    def forward(self, history):
        velocity, mu, logvar = self.encode(history)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return velocity, mu, logvar, self.decoder(torch.cat((velocity, z), -1))


class DreamActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([64, 512, 256, 128, 12])
        self.log_std = nn.Parameter(torch.zeros(12))

    def forward(self, features):
        return self.net(features)

    def distribution(self, features):
        return torch.distributions.Normal(self(features), self.log_std.clamp(-5.0, 2.0).exp())


class DreamPolicy(nn.Module):
    """Stateless export: caller maintains [batch, 6, 45] raw history."""

    def __init__(self, normalizer, cenet, actor):
        super().__init__()
        self.normalizer = copy.deepcopy(normalizer)
        self.cenet = copy.deepcopy(cenet)
        self.actor = copy.deepcopy(actor.net)

    def forward(self, history):
        torch._assert(history.dim() == 3, "Expected [batch,6,45] history")
        torch._assert(
            history.size(1) == 6 and history.size(2) == 45, "Expected 6 frames of 45 observations"
        )
        h = self.normalizer(history)
        velocity, mu, _ = self.cenet.encode(h)
        return self.actor(torch.cat((h[:, -1], velocity, mu), dim=-1))
