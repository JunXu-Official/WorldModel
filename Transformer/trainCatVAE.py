# 训练 CatVAE。
from torch.utils.data import TensorDataset, DataLoader
from config import USE_CUDA, USE_TPU, DEVICE, optimizer_step
import matplotlib.pyplot as plt
import torch
from gen_data import make_shape_images
from catVAE import CatVAE
from parameters import Parameter
import torch.nn.functional as F


def get_dataset(images, batch_size):
    dataset = TensorDataset(images)
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0 if USE_TPU else 2, pin_memory=USE_CUDA)
    return data_loader


def train(model, data_loader, epoches):
    
    opt_vae = torch.optim.Adam(model.parameters(), lr=3e-4)
    vae_losses = []
    print(f'正在训练...')
    for epoch in range(epoches):
        epoch_loss = 0.0
        for (batch,) in data_loader:
            batch = batch.to(DEVICE)
            recon, z, idx, logits = model(batch)
            # 重建损失
            recon_loss = F.mse_loss(recon, batch)
            # 熵正则化：鼓励均匀使用各类别
            probs = F.softmax(logits, dim=-1).mean(0)  # (K,)
            entropy_reg = (probs * (probs + 1e-8).log()).sum()  # 负熵
            loss = recon_loss + 0.01 * entropy_reg
            opt_vae.zero_grad()
            loss.backward()
            optimizer_step(opt_vae)
            epoch_loss += recon_loss.item()
        vae_losses.append(epoch_loss / len(data_loader))
        if (epoch + 1) % 10 == 0:
            print(f'轮次{epoch+1:3d}/{epoches} recon_loss={vae_losses[-1]:.4f}')

    print('CatVAE 训练完成。')
    plt.figure(figsize=(7, 3))
    plt.plot(vae_losses, linewidth=2, color='steelblue')
    plt.xlabel('轮次')
    plt.ylabel('重建损失（MSE）')
    plt.title('类别VAE：重建损失曲线')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def rollout(model, images):
    # 可视化 CatVAE 重建效果。
    model.eval()
    with torch.no_grad():
        sample = images[:8].to(DEVICE)
        recon, _, _, _ = model(sample)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(sample[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[0, i].set_title('原图' if i == 0 else '')
        axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        axes[1, i].set_title('重建' if i == 0 else '')
    plt.suptitle('CatVAE：原图（上）与重建（下）对比')
    plt.tight_layout()
    plt.show()
    model.train()

if __name__ == "__main__":
    # 超参数
    parameter = Parameter()
    # 生成图像数据集
    images = make_shape_images(
    n=parameter.n_img,
    size=parameter.batch_size, 
    seed=parameter.seed
)
    # 得到训练数据
    data_loader = get_dataset(
        images=images,
        batch_size=parameter.batch_size,
    )
    # 模型初始化
    catvae = CatVAE(num_categories=parameter.num_categories, tau=parameter.tau).to(DEVICE)

    train(
        model=catvae,
        data_loader=data_loader, 
        epoches=parameter.epoches
    )
    rollout(
        model=catvae,
        images=images
    )