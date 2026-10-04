"""PPO on frozen rollout features, followed by a separate CENet update."""

import torch

from dreamwaq_mjlab.learning.history import estimator_loss


def returns_advantages(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        running = delta + gamma * lam * (~dones[t]).float() * running
        advantages[t] = running
    return advantages + values, advantages


def update(
    actor,
    critic,
    cenet,
    optimizer,
    ce_optimizer,
    data,
    learning_rate,
    beta=1.0,
    epochs=5,
    minibatches=4,
):
    ret, adv = returns_advantages(
        data["rewards"], data["values"], data["next_values"], data["dones"]
    )
    flat = {k: v.flatten(0, 1) for k, v in data.items()}
    flat["returns"] = ret.flatten()
    flat["adv"] = (adv.flatten() - adv.mean()) / (adv.std() + 1e-8)
    size = flat["adv"].numel()
    losses = []
    last_kl = 0.0
    for _ in range(epochs):
        for ids in torch.randperm(size, device=adv.device).chunk(minibatches):
            dist = actor.distribution(flat["features"][ids])
            lp = dist.log_prob(flat["actions"][ids]).sum(-1)
            ratio = (lp - flat["log_probs"][ids]).exp()
            policy = -torch.minimum(
                ratio * flat["adv"][ids], ratio.clamp(0.8, 1.2) * flat["adv"][ids]
            ).mean()
            value = critic(flat["critic"][ids]).squeeze(-1)
            clipped = flat["values"][ids] + (value - flat["values"][ids]).clamp(-0.2, 0.2)
            vloss = torch.maximum(
                (value - flat["returns"][ids]).square(), (clipped - flat["returns"][ids]).square()
            ).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy + vloss - 0.01 * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), 1.0
            )
            optimizer.step()
            with torch.no_grad():
                oldmean = flat["means"][ids]
                oldstd = flat["stds"][ids]
                kl = (
                    (
                        torch.log(dist.scale / oldstd)
                        + (oldstd.square() + (oldmean - dist.loc).square())
                        / (2 * dist.scale.square())
                        - 0.5
                    )
                    .sum(-1)
                    .mean()
                )
                last_kl = float(kl)
            losses.append((float(policy.detach()), float(vloss.detach())))
    if last_kl > 0.02:
        learning_rate = max(1e-5, learning_rate / 1.5)
    elif 0 < last_kl < 0.005:
        learning_rate = min(1e-3, learning_rate * 1.5)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    ce_losses = []
    # Encoder remains fixed throughout PPO, including all minibatches.
    for ids in torch.randperm(size, device=adv.device).chunk(minibatches):
        loss, parts = estimator_loss(
            cenet,
            flat["history"][ids],
            flat["velocity"][ids],
            flat["next_clean"][ids],
            ~flat["dones"][ids],
            beta,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite CENet loss")
        ce_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cenet.parameters(), 1.0)
        ce_optimizer.step()
        ce_losses.append([float(x.detach()) for x in parts])
    return learning_rate, {
        "policy": sum(x[0] for x in losses) / len(losses),
        "value": sum(x[1] for x in losses) / len(losses),
        "kl": last_kl,
        "velocity_mse": sum(x[0] for x in ce_losses) / len(ce_losses),
        "reconstruction": sum(x[1] for x in ce_losses) / len(ce_losses),
        "vae_kl": sum(x[2] for x in ce_losses) / len(ce_losses),
    }
