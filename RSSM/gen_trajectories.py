import numpy as np
import torch
from config import DEVICE
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from VAE.model import Encoder, Decoder

def make_trajectory(T=20, img_size=64, seed=None):
    """
    生成合成轨迹数据：模拟一个彩色矩形在黑色背景上随机移动的过程
    """
    rng = np.random.RandomState(seed)   # 创建独立的随机数生成器
    color = rng.rand(3).astype(np.float32)  # 随机生成一个RGB颜色，3通道值属于[0,1]
    w  = rng.randint(10, 20)    #矩形的宽
    h  = rng.randint(10, 20)    # 矩形的高
    x  = float(rng.randint(0, img_size - w))    # 矩形左上角的初始坐标
    y  = float(rng.randint(0, img_size - h))
    vx = float(rng.randint(-3, 4))  # 矩形在X,Y轴上的基础速度
    vy = float(rng.randint(-3, 4))
    frames  = []        # 按序存储每一帧的图像数据
    actions = []        # 按序存储每一帧对应的动作
    for _ in range(T):
        # 生成纯黑背景图
        img = np.zeros((img_size, img_size, 3), dtype=np.float32)
        # 计算安全的绘制坐标
        x1, y1 = int(np.clip(x, 0, img_size - w)), int(np.clip(y, 0, img_size - h))
        # 在背景上绘制彩色矩形
        img[y1:y1 + h, x1:x1 + w] = color
        # 保存当前帧
        frames.append(img)
        # 生成随机动作
        actions.append(int(rng.randint(0, 2)))
        # 更新矩形坐标
        x = float(np.clip(x + vx + rng.uniform(-1, 1), 0, img_size - w))
        y = float(np.clip(y + vy + rng.uniform(-1, 1), 0, img_size - h))
    obs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # [T,3,H,W]  [20,3,64,64]
    act = torch.tensor(actions, dtype=torch.float32)               # [T]
    return {'obs': obs, 'actions': act}


def gen_trajs(t_steps, img_ch, img_size, latent_dim, n_traj, n_train):
    """
    生成一批合成估计，利用一个预训练好的编码器将高维的图像序列压缩到低维空间，并划分训练集和测试集
    """
    # 编码器
    encoder = Encoder(img_ch, latent_dim).to(DEVICE)
    print('正在生成 {n_traj} 条合成轨迹...')
    trajectories = [make_trajectory(T=t_steps, img_size=img_size, seed=i) for i in range(n_traj)]
    print(f"每条轨迹的观测形状:{trajectories[0]['obs'].shape}")
    print(f"每条轨迹的动作形状:{trajectories[0]['actions'].shape}")
    # 将观测编码为潜在序列 z [N, T, 32]
    print('正在编码观测...')
    latent_list = []
    with torch.no_grad():
        for traj in trajectories:
            obs = traj['obs'].to(DEVICE)      # [T,3,H,W]
            z = encoder.encode(obs)         # [T, latent_dim]
            latent_list.append(z.cpu())
    Z_all = torch.stack(latent_list, dim=0)                                     # [N,T,32]
    A_all = torch.stack([t['actions'] for t in trajectories], dim=0)            # [N,T]
    # 训练/测试划分
    Z_train, A_train = Z_all[:n_train].to(DEVICE), A_all[:n_train].to(DEVICE)
    Z_test,  A_test  = Z_all[n_train:].to(DEVICE), A_all[n_train:].to(DEVICE)
    print(f'Z_all 形状: {Z_all.shape}  (N, T, latent_dim)')
    print(f'训练集: {n_train} 条轨迹 | 测试集: {n_traj - n_train} 条轨迹')
    return Z_train, A_train, Z_test, A_test
   

if __name__ == '__main__':
    gen_trajs(20, 3, 64, 32, 200, 180)