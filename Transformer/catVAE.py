import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from parameters import Parameter

# 直通 Gumbel-softmax。
def straight_through_gumbel(logits, tau=1.0):
    """返回离散采样的直通（straight-through）估计器。"""
    y_soft = F.gumbel_softmax(logits, tau=tau, hard=False)
    y_hard = F.one_hot(y_soft.argmax(-1), num_classes=logits.shape[-1]).float()
    # 直通：前向使用 y_hard，反向梯度流经 y_soft
    return (y_hard - y_soft).detach() + y_soft


class CatVAEEncoder(nn.Module):
    """将 3x64x64 帧映射为 NUM_CATEGORIES 类别 logits 的 CNN。"""
    def __init__(self, num_categories):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 4x4
            nn.ReLU(),
            nn.Flatten(),                 # 256*4*4 = 4096
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_categories),
        )

    def forward(self, x):
        return self.net(x)  # (B, num_categories)


class CatVAEDecoder(nn.Module):
    """将 32 维 one-hot 嵌入解码回 3x64x64 的 MLP + ConvTranspose。"""
    def __init__(self, num_categories):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_categories, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 64x64
            nn.Sigmoid(),
        )

    def forward(self, z_onehot):
        h = self.fc(z_onehot)
        h = h.view(-1, 256, 4, 4)
        return self.deconv(h)  # (B, 3, 64, 64)


class CatVAE(nn.Module):
    def __init__(self, num_categories, tau=1.0):
        super().__init__()
        self.encoder = CatVAEEncoder(num_categories)
        self.decoder = CatVAEDecoder(num_categories)
        self.tau = tau

    def encode(self, x):
        """返回直通 one-hot 向量和 argmax 类别索引。"""
        logits = self.encoder(x)           # (B, K)
        z = straight_through_gumbel(logits, tau=self.tau)  # (B, K)
        idx = logits.argmax(-1)            # (B,)
        return z, idx, logits

    def forward(self, x):
        z, idx, logits = self.encode(x)
        recon = self.decoder(z)
        return recon, z, idx, logits


if __name__ == "__main__":
    from parameters import Parameter
    parameter = Parameter()
    catvae = CatVAE(num_categories=parameter.num_categories, tau=parameter.tau).to(DEVICE)
    total_params = sum(p.numel() for p in catvae.parameters())
    print(f'CatVAE 参数量：{total_params:,}')