import math

import torch


class History:
    def __init__(self, first):
        self.data = first[:, None, :].repeat(1, 6, 1)

    def append(self, obs):
        self.data = torch.cat((self.data[:, 1:], obs[:, None]), dim=1)

    def reset(self, ids, first):
        self.data[ids] = first[ids, None, :].repeat(1, 6, 1)


def adaboot_probability(rewards):
    if len(rewards) < 2:
        return 0.0
    mean = sum(rewards) / len(rewards)
    if not math.isfinite(mean) or mean <= 1e-8:
        return 0.0
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return max(0.0, min(1.0, 1.0 - math.tanh(math.sqrt(var) / mean)))


def estimator_loss(cenet, histories, velocity, next_obs, valid, beta=1.0):
    predicted, mu, logvar, reconstruction = cenet(histories)
    estimation = (predicted - velocity).square().mean()
    mask = valid[:, None].float()
    reconstruction_loss = ((reconstruction - next_obs).square() * mask).sum() / (
        mask.sum().clamp_min(1) * 45
    )
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    return estimation + reconstruction_loss + beta * kl, (estimation, reconstruction_loss, kl)
