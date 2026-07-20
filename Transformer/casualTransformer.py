import torch
import torch.nn as nn
import math
from parameters import Parameter
from config import DEVICE


# 因果 Transformer 世界模型。
class CausalTransformerWM(nn.Module):
    def __init__(self, num_categories, d_model, n_heads, n_layers, n_actions, max_len):
        super().__init__()
        self.d_model = d_model
        self.num_categories = num_categories
        # 将 z（one-hot）和动作投影到 d_model 维
        self.z_proj = nn.Linear(num_categories, d_model)
        self.a_embed = nn.Embedding(n_actions, d_model)
        # 位置编码
        pos = torch.arange(max_len * 2).unsqueeze(1)  # *2 用于 (z,a) 交错
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len * 2, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
        # 使用 nn.MultiheadAttention 的 Transformer 层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True, norm_first=True
            )
            for _ in range(n_layers)
        ])
        # 输出头
        self.token_head = nn.Linear(d_model, num_categories)   # 下一 token 预测
        self.reward_head = nn.Linear(d_model, 1)               # 奖励回归
        self.done_head = nn.Linear(d_model, 1)                 # 结束分类

    def _causal_mask(self, T):
        """上三角掩码：True 表示"忽略该位置"。"""
        return torch.triu(torch.ones(T, T, device=self.pe.device), diagonal=1).bool()

    def forward(self, z_seq, a_seq):
        """
        z_seq: (B, T, num_categories)  one-hot 离散潜在
        a_seq: (B, T)                  整数动作
        返回：
          token_logits: (B, T, num_categories)
          reward_pred:  (B, T, 1)
          done_pred:    (B, T, 1)
        """
        B, T, _ = z_seq.shape

        z_emb = self.z_proj(z_seq)          # (B, T, D)
        a_emb = self.a_embed(a_seq)         # (B, T, D)

        # 交错 z 和 a token：[z0, a0, z1, a1, ...] -> 长度为 2T
        tokens = torch.stack([z_emb, a_emb], dim=2).view(B, 2 * T, self.d_model)
        tokens = tokens + self.pe[:2 * T].unsqueeze(0)

        mask = self._causal_mask(2 * T)
        h = tokens
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=False)

        # 提取 z 位置（偶数索引）用于预测头
        z_h = h[:, 0::2, :]   # (B, T, D)  -- 每个 z 位置的输出

        token_logits = self.token_head(z_h)   # (B, T, K)
        reward_pred  = self.reward_head(z_h)  # (B, T, 1)
        done_pred    = self.done_head(z_h)    # (B, T, 1)
        return token_logits, reward_pred, done_pred


def main():
    parameter = Parameter()
    transformer_wm = CausalTransformerWM(
        num_categories=parameter.num_categories,
        d_model=parameter.d_model,
        n_heads=parameter.n_heads,
        n_layers=parameter.n_layers,
        n_actions=parameter.n_actions,
        max_len=parameter.seq_len
    ).to(DEVICE)
    total_params_t = sum(p.numel() for p in transformer_wm.parameters())
    print(f'因果 Transformer 参数量：{total_params_t:,}')


if __name__ == '__main__':
    main()