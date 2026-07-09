import torch
from utility import DEVICE
import torch.nn.functional as F
import torch.nn as nn


def lambda_returns(rewards, values, gamma=0.99, lam=0.95):
    """计算想象推演的 lambda 回报目标。"""
    H = rewards.shape[0]
    G = torch.zeros(H, device=DEVICE)
    G_next = values[H]
    for t in reversed(range(H)):
        td = rewards[t] + gamma * values[t + 1]
        G_next = (1 - lam) * td + lam * gamma * G_next
        G[t] = G_next
    return G


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def imagined_rollout(models, start_h, start_z, horizon, action_dim, differentiable=False, tau=0.8):
    """在想象空间中展开潜在轨迹。"""
    h, z = start_h, start_z
    h_seq, z_seq, r_seq, ent_seq = [], [], [], []

    for _ in range(horizon):
        logits = models['actor'](h, z)
        dist = torch.distributions.Categorical(logits=logits)
        ent_seq.append(dist.entropy().mean())

        if differentiable:
            a_oh = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            r_hat = models['reward_model'](h, z, a_oh)
            h_next = models['rssm'].step(h, z, a_oh)
            prior_out = models['rssm'].prior_net(h_next)
            prior_mu, _ = prior_out.chunk(2, dim=-1)
            z_next = prior_mu
        else:
            a = dist.sample()
            a_oh = F.one_hot(a, num_classes=action_dim).float()
            with torch.no_grad():
                r_hat = models['reward_model'](h, z, a_oh)
                h_next = models['rssm'].step(h, z, a_oh)
                prior_out = models['rssm'].prior_net(h_next)
                prior_mu, _ = prior_out.chunk(2, dim=-1)
                z_next = prior_mu

        h_seq.append(h)
        z_seq.append(z)
        r_seq.append(r_hat)
        h, z = h_next, z_next

    h_all = torch.stack(h_seq, dim=0)
    z_all = torch.stack(z_seq, dim=0)
    r_all = torch.stack(r_seq, dim=0).mean(dim=-1)
    return h_all, z_all, r_all, ent_seq


def behavior_update(models, opts, action_dim, start_h, start_z, horizon):
    """在想象回报上训练 Critic。"""
    models['critic'].train()
    models['reward_model'].eval()
    models['rssm'].eval()
    with torch.no_grad():
        h_all, z_all, r_all, ent_seq = imagined_rollout(
            models=models,
            start_h=start_h.detach(),
            start_z=start_z.detach(), 
            horizon=horizon, 
            action_dim=action_dim,
            differentiable=False)

    v_all = torch.zeros(horizon + 1, device=DEVICE)
    for t in range(horizon):
        v_all[t] = models['critic'](h_all[t], z_all[t]).mean().detach()
    v_all[horizon] = models['critic'](h_all[-1], z_all[-1]).mean().detach()
    G = lambda_returns(r_all, v_all)
    opts['opt_critic'].zero_grad()
    v_pred = torch.stack([models['critic'](h_all[t], z_all[t]).mean() for t in range(horizon)])
    critic_loss = F.mse_loss(v_pred, G.detach())
    critic_loss.backward()
    nn.utils.clip_grad_norm_(models['critic'].parameters(), 100.0)
    opts['opt_critic'].step()

    return torch.stack(ent_seq).mean().item()
