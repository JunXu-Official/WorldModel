import torch
import torch.nn as nn
import math
from config import DEVICE
import torch.nn.functional as F

class MDNRNN(nn.Module):
    """
        GRU+MDN 输出头，对z_(t+1)预测由3个高斯分量组成的混合分布。
        MDN 损失：混合分布的负对数似然。
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128, n_mix=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_mix = n_mix  # 模型未来的状态z_(t+1)可能由三种不同的情况组成
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        # logits (K), mu (K*D), log_sigma (K*D)
        # 描述一个由K个高斯分布组合而成的混合模型通常需要三类参数
        # 权重：描述K个分布各自占的概率
        # 均值：每个分布的中心，分布中心是D维向量，需要K*D个参数
        # 标准差的对数：K*D个参数。总参数量=K+K*D+K*D
        self.mdn_head = nn.Linear(hidden_dim, n_mix + 2 * n_mix * latent_dim)

    def _split(self, out):
        """
            参数切分。前K个是logits, 中间K*D是均值。最后的K*D是对数标准差
        """
        K, D = self.n_mix, self.latent_dim
        logits = out[..., :K]
        mu = out[..., K:K + K * D].reshape(*out.shape[:-1], K, D)
        log_s = out[..., K + K * D:].reshape(*out.shape[:-1], K, D)
        return logits, mu, log_s

    def forward(self, z_seq, a_seq):
        """
            z_seq: 输入状态，形状是[B, T, D]
            a_seq: 对应动作，形状是[B, T]
        """
        B, T, _ = z_seq.shape
        # 初始化隐状态
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        all_logits, all_mu, all_ls = [], [], []
        for t in range(T - 1):
            inp = torch.cat([z_seq[:, t], a_seq[:, t].unsqueeze(-1)], dim=-1)
            h = self.gru(inp, h)
            # 计算混合高斯分布的参数
            lg, mu, ls = self._split(self.mdn_head(h))
            all_logits.append(lg)
            all_mu.append(mu)
            all_ls.append(ls)
        return (
            torch.stack(all_logits, dim=1),     # [B, T-1, K]
            torch.stack(all_mu, dim=1),         # [B, T-1, K, D]
            torch.stack(all_ls, dim=1),         # [B, T-1, K, D]
        )

    def mdn_loss(self, logits, mu, log_sigma, target):
        """
            MDN_RNN的负对数似然损失
        """
        B, T, K, D = mu.shape
        tgt = target.unsqueeze(2).expand_as(mu)  # [B, T, 1, D]扩展为[B,T,K,D]
        # 计算真实值在每个高斯分量下的对数概率
        sigma = log_sigma.exp().clamp(min=1e-4)     # 把logstd转化为std
        log_p = -0.5 * (((tgt - mu) / sigma) ** 2 + 2 * log_sigma
                        + math.log(2 * math.pi))
        log_p  = log_p.sum(-1)                          # [B,T,K]
        log_pi = F.log_softmax(logits, dim=-1)          # [B,T,K]
        return -torch.logsumexp(log_pi + log_p, dim=-1).mean()

    def rollout(self, z0, a_seq):
        """
            rollout测试
        """
        z  = z0
        h  = torch.zeros(1, self.hidden_dim, device=z0.device)
        zs = [z]
        for a in a_seq:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h = self.gru(inp, h)
            lg, mu, _ = self._split(self.mdn_head(h))
            k = lg[0].argmax().item()
            z = mu[0, k].unsqueeze(0)   # [1, D]
            zs.append(z)
        return torch.stack(zs, dim=1)

if __name__ == '__main__':
    LATENT_DIM=32
    ACTION_DIM=1
    HIDDEN_DIM=128
    mdn_model = MDNRNN(LATENT_DIM, ACTION_DIM, HIDDEN_DIM, n_mix=3).to(DEVICE)
    print(f'MDNRNN 参数量: {sum(p.numel() for p in mdn_model.parameters()):,}')
