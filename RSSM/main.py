import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) 
import torch
from gru import GRUDynamics
from mdn_rnn import MDNRNN
from rssm import RSSM
import torch.nn as nn
import torch.nn.functional as F
from gen_trajectories import gen_trajs
from config import DEVICE
import numpy as np
from VAE.model import Decoder
import matplotlib.pyplot as plt
from IPython.display import display




def run_epoch(model, batch, optimizer, Z, A, loss_fn):
    model.train()
    N  = Z.shape[0]
    idx = torch.randperm(N)
    total, nb = 0.0, 0
    for s in range(0, N, batch):
        bi = idx[s:s + batch]
        zb, ab = Z[bi], A[bi]
        optimizer.zero_grad()
        loss = loss_fn(model, zb, ab)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); nb += 1
    return total / nb


def gru_loss(m, zb, ab):
    return F.mse_loss(m(zb, ab), zb[:, 1:])


def mdn_loss_fn(m, zb, ab):
    logits, mu, ls = m(zb, ab)
    return m.mdn_loss(logits, mu, ls, zb[:, 1:])


def rssm_loss(m, zb, ab):
    return m(zb, ab)


def train(epoch, batch, lr, z_train, a_train):
     # 实例化模型
    gru_model = GRUDynamics().to(DEVICE)
    mdn_model = MDNRNN().to(DEVICE)
    rssm_model = RSSM().to(DEVICE)
    opt_gru = torch.optim.Adam(gru_model.parameters(),  lr=LR)
    opt_mdn = torch.optim.Adam(mdn_model.parameters(),  lr=LR)
    opt_rssm = torch.optim.Adam(rssm_model.parameters(), lr=LR)
    # loss日志
    losses_gru, losses_mdn, losses_rssm = [], [], []
    print(f'训练 3 个模型，共 {epoch} 轮...')
    for epoch in range(1, epoch + 1):
        lg = run_epoch(gru_model, batch, opt_gru,  z_train, a_train, gru_loss)
        lm = run_epoch(mdn_model, batch, opt_mdn,  z_train, a_train, mdn_loss_fn)
        lr = run_epoch(rssm_model, batch, opt_rssm, z_train, a_train, rssm_loss)
        losses_gru.append(lg)
        losses_mdn.append(lm)
        losses_rssm.append(lr)
        if epoch % 5 == 0 or epoch == 1:
            print(f'第 {epoch:3d} 轮 | GRU: {lg:.4f} | MDN-RNN: {lm:.4f} | RSSM: {lr:.4f}')

    checkpoint = {
    'rssm_state_dict': rssm_model.state_dict(),
    'hidden_dim': 128,
    'latent_dim': 32,
    'action_dim': 1,
    'epochs_trained': EPOCHS,
    'final_loss': losses_rssm[-1],
}
    torch.save(checkpoint, 'rssm.pt')
    return [gru_model, mdn_model, rssm_model], [losses_gru, losses_mdn, losses_rssm]


def normalize(curve):
    # 归一化训练损失曲线。
    a = np.array(curve, dtype=np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)

def plott(loss_list):
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(1, EPOCHS + 1)
    ax.plot(xs, normalize(loss_list[0]),  label='GRU (MSE)',     color='tab:blue')
    ax.plot(xs, normalize(loss_list[1]),  label='MDN-RNN (NLL)', color='tab:orange')
    ax.plot(xs, normalize(loss_list[2]), label='RSSM (ELBO)',   color='tab:red')
    ax.set_xlabel('轮次')
    ax.set_ylabel('归一化损失 [0, 1]')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.show()


def test_rollout(model_list, decoder, rollout_steps, z_test, a_test):
    # 使用第一条测试轨迹。
    z_traj = z_test[0]           # [T, D]
    a_traj = a_test[0]           # [T]
    z0 = z_traj[0].unsqueeze(0)        # [1, D]
    a_seq = a_traj[:rollout_steps]        # [10]
    for model in model_list:
        model.eval()
    with torch.no_grad():
        zs_gru  = model_list[0].rollout(z0, a_seq).squeeze(0)    # [11, D]
        zs_mdn  = model_list[1].rollout(z0, a_seq).squeeze(0)
        zs_rssm = model_list[2].rollout(z0, a_seq).squeeze(0)

    def decode_seq(zs):
        """zs [S, D] -> numpy [S, H, W, 3]，值域 [0,1]。"""
        zs = zs.to(DEVICE)
        h = torch.zeros(zs.size(0), 128, device=DEVICE)
        imgs = decoder(zs, h)  # [S, 3, H, W]
        return imgs.detach().cpu().permute(0, 2, 3, 1).numpy()

    imgs_gru  = decode_seq(zs_gru)
    imgs_mdn  = decode_seq(zs_mdn)
    imgs_rssm = decode_seq(zs_rssm)
    imgs_gt   = decode_seq(z_traj[:rollout_steps + 1])   # 真实帧

    # 图像网格：真实帧、GRU、MDN-RNN、RSSM。
    N_COLS = rollout_steps + 1
    row_labels = ['真实帧', 'GRU', 'MDN-RNN', 'RSSM']
    row_imgs = [imgs_gt, imgs_gru, imgs_mdn, imgs_rssm]
    fig, axes = plt.subplots(
        4,
        N_COLS,
        figsize=(N_COLS * 1.7, 4.4),
        constrained_layout=True,
    )
    fig.patch.set_facecolor('white')
    for r, (label, imgs) in enumerate(zip(row_labels, row_imgs)):
        for c in range(N_COLS):
            ax = axes[r, c]
            ax.imshow(np.clip(imgs[c], 0, 1), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.set_ylabel(
                    label,
                    fontsize=9,
                    rotation=0,
                    labelpad=36,
                    va='center',
                    ha='right',
                )
            if r == 0:
                ax.set_title(f'步骤 {c}', fontsize=9, pad=8)
    fig.suptitle('10 步想象 Rollout与真实帧对比', fontsize=12, y=1.05)
    display(fig)
    plt.close(fig)
    return imgs_gru, imgs_mdn, imgs_rssm, imgs_gt


# 各步像素 MSE（相对于真实帧）。
def pixel_mse_per_step(pred, gt):
    return [float(((p - g) ** 2).mean()) for p, g in zip(pred, gt)]

def plott_rollout(n_cols, imgs_gru, imgs_mdn, imgs_rssm, imgs_gt):
    mse_gru  = pixel_mse_per_step(imgs_gru,  imgs_gt)
    mse_mdn  = pixel_mse_per_step(imgs_mdn,  imgs_gt)
    mse_rssm = pixel_mse_per_step(imgs_rssm, imgs_gt)
    steps_x  = list(range(n_cols))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_x, mse_gru,  marker='o', label='GRU',     color='tab:blue')
    ax.plot(steps_x, mse_mdn,  marker='s', label='MDN-RNN', color='tab:orange')
    ax.plot(steps_x, mse_rssm, marker='^', label='RSSM',    color='tab:green')
    ax.set_xlabel('Rollout 步骤')
    ax.set_ylabel('像素 MSE')
    ax.set_title('各步像素 MSE（相对于真实帧）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def cal_mae_mse(model_rollout_fn, Z, A, max_h=5):
    """计算测试集上步长 1..max_h 的平均潜在 MSE。
    model_rollout_fn(z0, a_seq) -> [1, steps+1, D]
    """
    N, T, D = Z.shape
    errs   = np.zeros(max_h)
    counts = np.zeros(max_h)
    with torch.no_grad():
        for i in range(N):
            for t0 in range(T - max_h):
                z0   = Z[i, t0].unsqueeze(0)          # [1, D]
                a_s  = A[i, t0:t0 + max_h]            # [max_h]
                zs   = model_rollout_fn(z0, a_s).squeeze(0)  # [max_h+1, D]
                for h in range(1, max_h + 1):
                    errs[h - 1]   += F.mse_loss(zs[h], Z[i, t0 + h]).item()
                    counts[h - 1] += 1
    return errs / counts


def plott_mae_mse(model_list, max_h):
    print('正在计算测试集上的步长误差...')
    for model in model_list:
        model.eval()
    err_gru  = cal_mae_mse(model_list[0].rollout,  Z_test, A_test, max_h)
    err_mdn  = cal_mae_mse(model_list[1].rollout,  Z_test, A_test, max_h)
    err_rssm = cal_mae_mse(model_list[2].rollout, Z_test, A_test, max_h)
    horizons = list(range(1, max_h + 1))
    print('\n各步长潜在 MSE:')
    print(f'{"步长":>8}  {"GRU":>10}  {"MDN-RNN":>10}  {"RSSM":>10}')
    for h, (eg, em, er) in enumerate(zip(err_gru, err_mdn, err_rssm), start=1):
        print(f'{h:>8}  {eg:>10.4f}  {em:>10.4f}  {er:>10.4f}')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(horizons, err_gru,  marker='o', label='GRU',     color='tab:blue')
    ax.plot(horizons, err_mdn,  marker='s', label='MDN-RNN', color='tab:orange')
    ax.plot(horizons, err_rssm, marker='^', label='RSSM',    color='tab:green')
    ax.set_xlabel('预测步长（步）')
    ax.set_ylabel('平均潜在 MSE')
    ax.set_title('单步至多步预测误差（保留测试集）')
    ax.set_xticks(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # 训练参数
    EPOCHS = 50
    BATCH  = 32
    LR = 1e-3
    t_steps = 20
    img_ch = 3
    img_size = 64
    latent_dim = 32
    n_trajs = 200
    n_train_trajs = 180
    rollout_steps = 10
    # 数据集加载
    Z_train, A_train, Z_test, A_test = gen_trajs(t_steps, img_ch, img_size, latent_dim, n_trajs, 
n_train_trajs)
    # 训练
    model_list, loss_list = train(
        epoch=EPOCHS,
        batch=BATCH,
        lr=LR,
        z_train=Z_train,
        a_train=A_train
    )
    plott(loss_list=loss_list)
    decoder = Decoder(img_ch=3, latent_dim=32, hidden_dim=128).to(DEVICE)
    imgs_gru, imgs_mdn, imgs_rssm, imgs_gt = test_rollout(
        model_list=model_list,
        decoder=decoder,
        rollout_steps=rollout_steps,
        z_test=Z_test,
        a_test=A_test
    )

    plott_rollout(
        n_cols=rollout_steps + 1,
        imgs_gru=imgs_gru,
        imgs_mdn=imgs_mdn,
        imgs_rssm=imgs_rssm,
        imgs_gt=imgs_gt
    )

    plott_mae_mse(model_list=model_list, max_h=5)