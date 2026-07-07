import torch
import torch.nn as nn
from config import DEVICE

class GRUDynamics(nn.Module):
    """GRU 动力学模型，hidden_dim=128，输入为 (latent_dim+action_dim)，输出为 latent_dim。
    接受 (z_t, a_t) 作为输入，经 GRU 更新后预测 z_{t+1}。
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_seq, a_seq):
        """z_seq [B,T,D], a_seq [B,T] -> pred_z [B,T-1,D]，对应步骤 1..T。"""
        B, T, _ = z_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=z_seq.device)
        preds = []
        for t in range(T - 1):
            inp = torch.cat([z_seq[:, t], a_seq[:, t].unsqueeze(-1)], dim=-1)
            h   = self.gru(inp, h)
            preds.append(self.output(h))
        return torch.stack(preds, dim=1)  # [B, T-1, D]

    def rollout(self, z0, a_seq):
        """从 z0 [1,D] 开始，按 a_seq 进行开环 rollout。
        返回 [1, steps+1, D]（包含 z0）。
        """
        z = z0
        h = torch.zeros(1, self.hidden_dim, device=z0.device)
        zs = [z]
        for a in a_seq:
            inp = torch.cat([z, a.view(1, 1)], dim=-1)
            h   = self.gru(inp, h)
            z   = self.output(h)
            zs.append(z)
        return torch.stack(zs, dim=1)  # [1, steps+1, D]

if __name__ == "__main__":
    LATENT_DIM=32
    ACTION_DIM=1
    HIDDEN_DIM=128
    gru_model = GRUDynamics(LATENT_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
    print(f'GRUDynamics 参数量: {sum(p.numel() for p in gru_model.parameters()):,}')