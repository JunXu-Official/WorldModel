import torch
from parameters import Parameter
import torch.nn as nn


class Actor(nn.Module):
    """将潜在状态 (h, z) 和轻量级观测特征映射为动作 logits。"""
    def __init__(self, parameter, obs_feat_dim=1):
        super().__init__()
        self.parameter = parameter
        inp = self.parameter.hidden_dim + self.parameter.latent_dim + obs_feat_dim
        self.net = nn.Sequential(
            nn.Linear(inp, self.parameter.ac_hidden),
            nn.ELU(),
            nn.Linear(self.parameter.ac_hidden, self.parameter.ac_hidden),
            nn.ELU(),
            nn.Linear(self.parameter.ac_hidden, self.parameter.action_dim),
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
    def __init__(self, parameter):
        super().__init__()
        self.parameter = parameter
        inp = self.parameter.hidden_dim + self.parameter.latent_dim
        self.net = nn.Sequential(
            nn.Linear(inp, self.parameter.ac_hidden),
            nn.ELU(),
            nn.Linear(self.parameter.ac_hidden, self.parameter.ac_hidden),
            nn.ELU(),
            nn.Linear(self.parameter.ac_hidden, 1),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class RewardModel(nn.Module):
    """从潜在状态和动作预测即时奖励。"""
    def __init__(self, parameter):
        super().__init__()
        self.parameter = parameter
        inp = self.parameter.hidden_dim + self.parameter.latent_dim + self.parameter.action_dim
        self.net = nn.Sequential(
            nn.Linear(inp, self.parameter.c_hidden),
            nn.ELU(),
            nn.Linear(self.parameter.ac_hidden, self.parameter.ac_hidden),
            nn.ELU(),
            nn.Linear(self.parameter.ac_hidden, 1),
        )

    def forward(self, h, z, a):
        return self.net(torch.cat([h, z, a], dim=-1)).squeeze(-1)