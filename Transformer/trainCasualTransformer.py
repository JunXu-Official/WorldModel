import numpy as np
import torch
from catVAE import CatVAE
from config import DEVICE
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import time
from catVAE import CatVAE
from casualTransformer import CausalTransformerWM
from parameters import Parameter
import matplotlib.pyplot as plt




# 合成轨迹数据。
def make_obs(cx, cy, size=64):
    img = np.zeros((3, size, size), dtype=np.float32)
    r = 8
    color = np.array([0.9, 0.3, 0.3], dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                img[:, y, x] = color
    return img


def generate_trajectories(n_traj=200, horizon=20, size=64, seed=0):
    rng = np.random.RandomState(seed)
    obs_list, act_list, rew_list, done_list = [], [], [], []
    for _ in range(n_traj):
        cx = rng.randint(20, size - 20)
        cy = rng.randint(20, size - 20)
        traj_obs, traj_act, traj_rew, traj_done = [], [], [], []
        for t in range(horizon):
            traj_obs.append(make_obs(cx, cy, size))
            action = rng.randint(0, 2)
            traj_act.append(action)
            # 动作 0：向右移动，动作 1：向左移动
            cx = np.clip(cx + (4 if action == 0 else -4), 10, size - 10)
            rew = 1.0 if cx > size // 2 else 0.0
            traj_rew.append(rew)
            traj_done.append(0.0)
        obs_list.append(traj_obs)
        act_list.append(traj_act)
        rew_list.append(traj_rew)
        done_list.append(traj_done)
    obs_arr  = torch.tensor(np.array(obs_list),  dtype=torch.float32)   # (N, T, 3, 64, 64)
    act_arr  = torch.tensor(np.array(act_list),  dtype=torch.long)       # (N, T)
    rew_arr  = torch.tensor(np.array(rew_list),  dtype=torch.float32)   # (N, T)
    done_arr = torch.tensor(np.array(done_list), dtype=torch.float32)   # (N, T)
    return obs_arr, act_arr, rew_arr, done_arr


def embedding_catvae(model, obs, num_categories):
    """
        用CatVAE对所有obs进行编码
    """
    model.eval()
    N, T, C, H, W = obs.shape
    z_encoded = torch.zeros(N, T, num_categories)  # one-hot token

    with torch.no_grad():
        flat_obs = obs.view(N * T, C, H, W).to(DEVICE)
        logits_all = model.encoder(flat_obs)                              # (N*T, K)
        idx_all = logits_all.argmax(-1)                                    # (N*T,)
        z_onehot_all = F.one_hot(idx_all, num_classes=num_categories).float()  # (N*T, K)
        z_encoded = z_onehot_all.view(N, T, num_categories).cpu()

    print('编码后潜在张量形状：', z_encoded.shape)
    unique_tokens = idx_all.unique().numel()
    print(f'实际使用的类别 token 数：{unique_tokens} / {num_categories}')
    model.train()
    return z_encoded

def trainCasualTransformer(model, z_encoded, act, reward, done, num_categories, epoches, batch_size, lr):
    """
        训练transformer在潜在序列上学习时间动力学
    """
    traj_dataset = TensorDataset(z_encoded, act, reward, done)
    traj_loader  = DataLoader(traj_dataset, batch_size=batch_size, shuffle=True)
    opt_t = torch.optim.Adam(model.parameters(), lr=lr)
    token_losses, reward_losses = [], []
    epoch_times = []   # 每轮实际耗时（秒）
    print('正在训练因果 Transformer...')
    for epoch in range(epoches):
        t0 = time.time()
        ep_tok, ep_rew = 0.0, 0.0
        for z_b, a_b, r_b, d_b in traj_loader:
            z_b = z_b.to(DEVICE)
            a_b = a_b.to(DEVICE)
            r_b = r_b.to(DEVICE)
            d_b = d_b.to(DEVICE)
            # 预测：在位置 t 预测 token t+1、奖励 t、结束标志 t
            token_logits, reward_pred, done_pred = model(z_b, a_b)  # (B, T, K/1/1)
            # 下一 token 标签：偏移 1 位，忽略最后一个位置
            target_idx = z_b[:, 1:, :].argmax(-1)           # (B, T-1)
            pred_logits = token_logits[:, :-1, :]            # (B, T-1, K)
            tok_loss = F.cross_entropy(
                pred_logits.reshape(-1, num_categories),
                target_idx.reshape(-1)
            )
            # 所有位置的奖励预测
            rew_loss = F.mse_loss(reward_pred.squeeze(-1), r_b)
            # 结束标志预测
            done_loss = F.binary_cross_entropy_with_logits(done_pred.squeeze(-1), d_b)
            loss = tok_loss + 0.5 * rew_loss + 0.1 * done_loss
            opt_t.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_t.step()
            ep_tok += tok_loss.item()
            ep_rew += rew_loss.item()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        token_losses.append(ep_tok / len(traj_loader))
        reward_losses.append(ep_rew / len(traj_loader))
        if (epoch + 1) % 5 == 0:
            print(f'轮次 {epoch+1:2d}/{epoches}  '
                f'tok_loss={token_losses[-1]:.4f}  '
                f'rew_loss={reward_losses[-1]:.4f}  '
                f'耗时={elapsed:.2f}s')
    print('Transformer 训练完成。')
    # --- 绘制 token 损失和奖励损失 ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(token_losses,  label='Token 预测损失（交叉熵）',  color='steelblue',  linewidth=2)
    ax.plot(reward_losses, label='奖励预测损失（MSE）', color='darkorange', linewidth=2)
    ax.set_xlabel('轮次')
    ax.set_ylabel('损失')
    ax.set_title('因果 Transformer：训练损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 参数初始化
    parameter = Parameter()
    # catVAE模型初始化
    catvae = CatVAE(num_categories=parameter.num_categories, tau=parameter.tau).to(DEVICE)
    # casualTransformer模型初始化
    causalTransformerWM = CausalTransformerWM(
        num_categories=parameter.num_categories,
        d_model=parameter.d_model,
        n_heads=parameter.n_heads,
        n_layers=parameter.n_layers,
        n_actions=parameter.n_actions,
        max_len=parameter.seq_len
    ).to(DEVICE)
    # 生成轨迹
    print('正在生成 200 条合成轨迹（每条 20 步）...')
    obs_arr, act_arr, rew_arr, done_arr = generate_trajectories(n_traj=parameter.n_traj, horizon=parameter.seq_len)
    print(f'obs: {obs_arr.shape}, act: {act_arr.shape}, rew: {rew_arr.shape}')
    # embedding编码
    z_encoded = embedding_catvae(catvae, obs_arr, parameter.num_categories)
    # 训练
    trainCasualTransformer(
        model=causalTransformerWM,
        z_encoded=z_encoded,
        act=act_arr,
        reward=rew_arr,
        done=done_arr,
        num_categories=parameter.num_categories,
        epoches=parameter.transformer_epoches,
        batch_size=parameter.transformer_batch_size,
        lr=parameter.transformer_lr
    )
