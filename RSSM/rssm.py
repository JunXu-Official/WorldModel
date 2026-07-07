import torch
import torch.nn as nn
from config import DEVICE
import torch.nn.functional as F

class RSSM(nn.Module):
    """循环状态空间模型（RSSM）。
    确定性路径：h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
    随机先验：   z_t ~ N(mu_prior(h_t), sigma_prior(h_t))
    随机后验：   z_t ~ N(mu_post(h_t, o_t), sigma_post(h_t, o_t))
    训练目标：   ELBO = 重建损失 + KL(后验 || 先验)
    hidden_dim=128, latent_dim=32
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 确定性循环
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # 先验网络：h_t -> (mu, logvar)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # 后验网络：(h_t, o_t) -> (mu, logvar)，o_t 为编码后的观测
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

        # 重建头：z -> 预测 z（潜在重建目标）
        self.recon = nn.Linear(latent_dim, latent_dim)

    def _rsample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, z_seq, a_seq):
        """计算整条轨迹的 ELBO 损失。
        z_seq [B,T,D], a_seq [B,T]。
        返回标量 ELBO 损失。
        """
        B, T, D = z_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        z = torch.zeros(B, D, device=z_seq.device)
        recon_loss = z_seq.new_zeros(())
        kl_loss    = z_seq.new_zeros(())
        for t in range(T):
            inp = torch.cat([z, a_seq[:, t].unsqueeze(-1)], dim=-1)
            h   = self.gru(inp, h)

            # 先验
            pr   = self.prior_net(h)
            mu_pr, lv_pr = pr.chunk(2, dim=-1)

            # 以观测潜在向量 o_t = z_seq[:, t] 为条件的后验
            po   = self.post_net(torch.cat([h, z_seq[:, t]], dim=-1))
            mu_po, lv_po = po.chunk(2, dim=-1)

            z = self._rsample(mu_po, lv_po)

            # 重建损失：预测观测潜在向量
            recon_loss = recon_loss + F.mse_loss(self.recon(z), z_seq[:, t])

            # KL(后验 || 先验)
            kl = 0.5 * (
                lv_pr - lv_po
                + (lv_po.exp() + (mu_po - mu_pr) ** 2) / lv_pr.exp().clamp(min=1e-4)
                - 1
            )
            kl_loss = kl_loss + kl.mean()

        return (recon_loss + kl_loss) / T

    def rollout(self, z0, a_seq):
        """仅使用先验（不观测）进行开环 rollout。
        返回 [1, steps+1, D]。
        """
        z  = z0
        h  = torch.zeros(1, self.hidden_dim, device=z0.device)
        zs = [z]
        for a in a_seq:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h   = self.gru(inp, h)
            pr  = self.prior_net(h)
            mu, _ = pr.chunk(2, dim=-1)
            z  = mu  # 确定性 rollout 使用先验均值
            zs.append(z)
        return torch.stack(zs, dim=1)

if __name__ == "__main__":
    LATENT_DIM=32
    ACTION_DIM=1
    HIDDEN_DIM=128
    rssm_model = RSSM(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
    print(f'RSSM 参数量: {sum(p.numel() for p in rssm_model.parameters()):,}')