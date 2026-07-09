import torch
import torch.nn as nn


class Actor(nn.Module):
    """将潜在状态 (h, z) 和轻量级观测特征映射为动作 logits。"""
    def __init__(self, latent_dim, action_dim, hidden_dim, ac_hidden_dim, obs_feat_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.ac_hidden_dim = ac_hidden_dim
        self.obs_feat_dim = obs_feat_dim
        inp = self.hidden_dim + self.latent_dim + self.obs_feat_dim
        self.net = nn.Sequential(
            nn.Linear(inp, self.ac_hidden_dim),
            nn.ELU(),
            nn.Linear(self.ac_hidden_dim, self.ac_hidden_dim),
            nn.ELU(),
            nn.Linear(self.ac_hidden_dim, self.action_dim),
        )

    def forward(self, h, z, bar_pos=None):
        if bar_pos is None:
            bar_pos = torch.zeros(h.shape[0], 1, device=h.device)
        logits = self.net(torch.cat([h, z, bar_pos], dim=-1))
        return logits

    def sample(self, h, z, bar_pos=None):
        logits = self.forward(h, z, bar_pos=bar_pos)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist


class Critic(nn.Module):
    """将潜在状态 (h, z) 映射为标量价值估计。"""
    def __init__(self, latent_dim, hidden_dim, ac_hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ac_hidden_dim = ac_hidden_dim
        inp = self.hidden_dim + self.latent_dim
        self.net = nn.Sequential(
            nn.Linear(inp, self.ac_hidden_dim),
            nn.ELU(),
            nn.Linear(self.ac_hidden_dim, self.ac_hidden_dim),
            nn.ELU(),
            nn.Linear(self.ac_hidden_dim, 1),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class RewardModel(nn.Module):
    """从潜在状态和动作预测即时奖励。"""
    def __init__(self, latent_dim, hidden_dim, action_dim, ac_hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ac_hidden_dim = ac_hidden_dim
        inp = self.hidden_dim + self.latent_dim + self.action_dim
        self.net = nn.Sequential(
            nn.Linear(inp, self.ac_hidden_dim),
            nn.ELU(),
            nn.Linear(self.ac_hidden_dim, self.ac_hidden_dim),
            nn.ELU(),
            nn.Linear(self.ac_hidden_dim, 1),
        )

    def forward(self, h, z, a):
        return self.net(torch.cat([h, z, a], dim=-1)).squeeze(-1)