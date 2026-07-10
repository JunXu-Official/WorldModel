import torch
import torch.nn as nn
from utility import DEVICE
import torch.nn.functional as F



class RSSM(nn.Module):
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.gru = nn.GRUCell(self.latent_dim + 1, self.hidden_dim)
        # 先验：p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )
        # 后验：q(z_t | h_t, e_t)
        self.post_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.latent_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )
        self.recon = nn.Linear(self.latent_dim, self.latent_dim)

    def _action_feature(self, action):
        """
            将输入的action强行变成[batchs-size, 1]的形状，并确保是浮点数
        """
        if action.dim() == 0:
            action = action.view(1, 1)
        elif action.dim() == 1:
            action = action.unsqueeze(-1)
        if action.shape[-1] > 1:
            action = action[..., 1:2]
        return action.float()

    def initial_state(self, batch_size):
        """
        初始化隐状态
        """
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        z = torch.zeros(batch_size, self.latent_dim, device=DEVICE)
        return h, z

    def prior(self, h):
        """
            RSSM先验
        """
        out = self.prior_net(h)
        mu, logvar = out.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1
        z = mu + std * torch.randn_like(std)
        return z, mu, std

    def posterior(self, h, enc_z):
        """
            RSSM后验
        """
        out = self.post_net(torch.cat([h, enc_z], dim=-1))
        mu, logvar = out.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1
        z = mu + std * torch.randn_like(std)
        return z, mu, std

    def step(self, h, z, action_onehot):
        """推进确定性状态，返回新的 h。"""
        action_feat = self._action_feature(action_onehot)
        inp = torch.cat([z, action_feat], dim=-1)
        h_new = self.gru(inp, h)
        return h_new
    
