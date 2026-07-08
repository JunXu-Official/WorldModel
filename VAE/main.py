from dataset import ShapeDataset
from torch.utils.data import Dataset, DataLoader
from config import USE_CUDA, DEVICE, optimizer_step
from model import VAE, elbo_loss
import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def train(model, LATENT_DIM, IMG_SIZE, IMG_CH, dataloader):
    # 训练参数、
    EPOCHES = 30
    lr = 1e-3
    kl_weight = 3.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    scaler = torch.amp.GradScaler('cuda', enabled=USE_CUDA)
    # 日志记录
    history_recon = []
    history_kl = []
    model.train()
    for epoch in range(1, EPOCHES+1):
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0
        for batch in dataloader:
            batch = batch.to(DEVICE, non_blocking=USE_CUDA)
            optimizer.zero_grad(True)
            with torch.amp.autocast('cuda', enabled=USE_CUDA):
                batch_recon, mu, log_var = model(batch)
                loss, recon_loss, kl_loss = elbo_loss(
                    img_ch=IMG_CH,
                    img_size=IMG_SIZE,
                    recon_x=batch_recon,
                    x=batch,
                    mu=mu,
                    log_var=log_var,
                    kl_weight=kl_weight
                )
            # 缩放loss并反向传播
            scaler.scale(loss).backward()
            optimizer_step(optimizer, scaler)
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1
        
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        history_recon.append(avg_recon)
        history_kl.append(avg_kl)
        # 打印训练日志
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch: {epoch}/{EPOCHES}, Avg_recon: {avg_recon}, KL: {avg_kl}")

    epochs_range = range(1, EPOCHES + 1)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    color_recon = '#2196F3'
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('重建损失（MSE）', color=color_recon)
    ax1.plot(epochs_range, history_recon, color=color_recon, linewidth=2, label='重建损失')
    ax1.tick_params(axis='y', labelcolor=color_recon)
    ax2 = ax1.twinx()
    color_kl = '#F44336'
    ax2.set_ylabel('KL 散度', color=color_kl)
    ax2.plot(epochs_range, history_kl, color=color_kl, linewidth=2, linestyle='--', label='KL 散度')
    ax2.tick_params(axis='y', labelcolor=color_kl)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('VAE 训练：重建损失与 KL 散度')
    fig.tight_layout()
    plt.show()

    print(f'最终重建损失: {history_recon[-1]:.5f}')
    print(f'最终KL散度: {history_kl[-1]:.6f}')
    checkpoint_path = 'vae_encoder.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder':          model.encoder.state_dict(),
        'decoder':          model.decoder.state_dict(),
        'latent_dim':       LATENT_DIM,
        'img_size':         IMG_SIZE,
        'img_channels':     IMG_CH,
        'final_recon_loss': history_recon[-1],
        'final_kl_loss':    history_kl[-1],
        'epochs_trained':   EPOCHES,
        'checkpoint_format': 'vae-v2',
    }, checkpoint_path)
    size_kb = os.path.getsize(checkpoint_path) / 1024
    print(f'权重文件已保存至: {checkpoint_path}  ({size_kb:.1f} KB)')

def test(model):
    # 视觉对比：保留批次的原始图像与重建图像
    model.eval()
    with torch.no_grad():
        sample = dataset[200:208].to(DEVICE)
        recon, _, _ = model(sample)
    n_show = 8
    fig, axes = plt.subplots(2, n_show, figsize=(16, 4))
    for i in range(n_show):
        axes[0, i].imshow(sample[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
    axes[0, 0].set_title('原始图像', loc='left', fontsize=12)
    axes[1, 0].set_title('重建图像', loc='left', fontsize=12)
    plt.suptitle('原始图像与重建图像对比（训练 30 轮后）', y=1.02)
    plt.tight_layout()
    plt.show()

def visulize_latent_space(model):
    model.eval()
    # 需要扫描的维度及扫描取值
    dims_to_vary = [0, 1, 2]
    sweep_values = np.linspace(-2, 2, 5)
    n_rows = len(dims_to_vary)
    n_cols = len(sweep_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2 + 0.5))
    with torch.no_grad():
        for row_idx, dim in enumerate(dims_to_vary):
            for col_idx, val in enumerate(sweep_values):
                z = torch.zeros(1, LATENT_DIM, device=DEVICE)
                z[0, dim] = float(val)
                decoded = model.decode(z)  # 形状: (1, 3, 64, 64)
                img = decoded[0].cpu().permute(1, 2, 0).numpy()  # (64, 64, 3)
                ax = axes[row_idx, col_idx]
                ax.imshow(img)
                ax.axis('off')
                if col_idx == 0:
                    ax.set_ylabel(f'维度 {dim}', fontsize=10, rotation=0, labelpad=30, va='center')
    # 列标题
    for col_idx, val in enumerate(sweep_values):
        axes[0, col_idx].set_title(f'z={val:.1f}', fontsize=9)
    plt.suptitle('潜在空间遍历：每次改变一个维度', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

def sample_in_latent_space(model):
    # 从学习到的先验中采样。
    model.eval()
    torch.manual_seed(42)

    n_samples = 16
    with torch.no_grad():
        z_random = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
        samples  = model.decode(z_random)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i].cpu().permute(1, 2, 0).numpy())
        ax.axis('off')

    plt.suptitle('从学习到的潜在空间随机采样（z ~ N(0, I)）', fontsize=12)
    plt.tight_layout()
    plt.show()

def save():
    # print('在下游项目中加载的方法：')
    # print("  ckpt = torch.load('vae_encoder.pt', map_location='cpu')")
    # print("  model = VAE(latent_dim=ckpt['latent_dim'])")
    # print("  model.load_state_dict(ckpt['model_state_dict'])")
    pass


if __name__ == '__main__':
    LATENT_DIM=32
    IMG_SIZE=64
    IMG_CH=3
    model = VAE(img_ch=IMG_CH, latent_dim=LATENT_DIM).to(DEVICE)
    dataset = ShapeDataset(n_samples=1000)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2 if USE_CUDA else 0,
        pin_memory=USE_CUDA,
    )
    params = sum(p.numel() for p in model.parameters())
    print('number of params:', params)

    train(model=model, LATENT_DIM=LATENT_DIM, IMG_SIZE=IMG_SIZE, IMG_CH=IMG_CH, dataloader=dataloader)
    # test(model=model)
    # visulize_latent_space(model=model)
    # sample_in_latent_space(model=model)