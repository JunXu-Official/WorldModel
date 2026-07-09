import torch
import torch.nn.functional as F
import torch.nn as nn




class Encoder(nn.Module):
    """将 3x64x64 图像编码为潜在均值和对数方差。"""

    def __init__(self, img_ch, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            # output_size = 下取整[(input_size +2 * padding - kernel_size) / stride] + 1
            # 3*64*64 --> 32*32*32
            nn.Conv2d(img_ch, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 32*32*32 --> 64*16*16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 64*16*16 --> 128*8*8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 128*8*8 --> 256*4*4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flat_dim = 256 * 4 * 4    # 4096
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        # [B, 4096]
        h = self.conv(x).flatten(start_dim=1)
        # [B, 32]
        return self.fc_mu(h), self.fc_log_var(h)
    
    def encode(self, x):
        h = self.conv(x).reshape(x.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_log_var(h)
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std), mu, logvar


class Decoder(nn.Module):
    """将潜在向量解码回 3x64x64 图像。"""

    def __init__(self, img_ch, latent_dim, hidden_dim):
        super().__init__()
        self.flat_dim = 256 * 4 * 4
        self.fc = nn.Linear(latent_dim + hidden_dim, self.flat_dim)
        self.deconv = nn.Sequential(
            # 256*4*4 --> 128*8*8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 128*8*8 --> 64*16*16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 64*16*16 --> 32*32*32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 32*32*32 --> 3*64*64
            nn.ConvTranspose2d(32, img_ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, h):
        x = self.fc(torch.cat([z, h], dim=-1))
        x = x.view(x.size(0), 256, 4, 4)
        return self.deconv(x)


class VAE(nn.Module):
    """结合编码器与解码器的变分自编码器。"""

    def __init__(self, img_ch, latent_dim):
        super().__init__()
        self.encoder = Encoder(img_ch, latent_dim)
        self.decoder = Decoder(img_ch, latent_dim, hidden_dim=128)

    def reparameterize(self, mu, log_var):
        """
            VAE不是直接输出一个隐变量而是输出一个概率分布N(μ,log_var**2)然后从这个分布中采样Z，给到decoder去重建图像
            重参数化
            采样 z = mu + sigma * epsilon。
        """
        std = torch.exp(0.5 * log_var)  # 标准差
        eps = torch.randn_like(std)     # 生成随机噪声
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        h = torch.zeros(z.size(0), 128, device=z.device)
        recon = self.decoder(z, h)
        return recon, mu, log_var

    def encode(self, x):
        """返回潜在均值，不进行采样。"""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        h = torch.zeros(z.size(0), 128, device=z.device)
        return self.decoder(z, h)

def elbo_loss(img_ch, img_size, recon_x, x, mu, log_var, kl_weight=1.0):
    """
    ELBO 损失 = 重建损失 + KL 散度。

    返回值：
        total_loss : 标量张量（用于反向传播）
        recon_loss : 标量张量（用于日志记录）
        kl_loss    : 标量张量（用于日志记录）
    """
    # 均方重建误差。
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # 对角高斯后验的解析形式 KL 散度。
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )
    # 对 KL 进行缩放，使其与重建项量级相当。
    kl_loss = kl_loss / (img_ch * img_size * img_size)
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss