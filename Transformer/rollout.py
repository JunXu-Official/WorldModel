import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, PATH
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from parameters import Parameter
import numpy as np


# PSNR 工具函数。
def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def load_rssm(rssm_path, latent_dim, action_dim, hidden_dim):
    """
        加载rssm模型
    """
    state = torch.load(rssm_path, map_location=DEVICE)
    from RSSM.rssm import RSSM
    rssm = RSSM(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim).to(DEVICE)
    if isinstance(state, dict) and 'rssm_state_dict' in state:
        rssm.load_state_dict(state['rssm_state_dict'])
        print(f"已从 {rssm_path} 加载 RSSM 权重 ")
    elif isinstance(state, dict) and 'rssm' in state:
        rssm.load_state_dict(state['rssm'])
        print(f'已从 {rssm_path} 加载 RSSM 权重（旧版 rssm 键）')
    else:
        rssm.load_state_dict(state)
    rssm.eval()
    # transformer_wm.eval()
    # catvae.eval()


def rollout(rssm_model, catvae_model, transformer_model, obs, act, z_encoded, n_eval, rollout_len, num_categories):
    """
    """
    # 使用前 N_EVAL 条轨迹作为评估集
    eval_obs  = obs[:n_eval].to(DEVICE)   # (N_EVAL, T, 3, 64, 64)
    eval_act  = act[:n_eval].to(DEVICE)   # (N_EVAL, T)
    eval_z    = z_encoded[:n_eval].to(DEVICE) # (N_EVAL, T, K)
    horizons = [1, 3, 5, 10]
    psnr_rssm_all  = {h: [] for h in horizons}
    psnr_trans_all = {h: [] for h in horizons}
    with torch.no_grad():
        for traj_i in range(n_eval):
            # 初始状态：编码第 0 步
            z0 = eval_z[traj_i, 0:1]         # (1, K)  one-hot
            acts = eval_act[traj_i]           # (T,)
            # ---- RSSM rollout ----
            z = z0.clone()
            h = torch.zeros(1, rssm_model.hidden_dim, device=DEVICE)
            rssm_preds = []  # 每步解码后的帧
            for t in range(rollout_len):
                a_t = acts[t:t+1].float().unsqueeze(-1)  # (1, 1)
                z, h = rssm_model.prior_step(z, h, a_t)
                frame = catvae_model.decoder(z)      # (1, 3, 64, 64)
                rssm_preds.append(frame)
            # ---- Transformer rollout ----
            # 以观测 z0 为起点，自回归预测
            z_seq = z0.unsqueeze(0)            # (1, 1, K)
            trans_preds = []
            for t in range(rollout_len):
                current_len = z_seq.shape[1]
                a_prefix = acts[:current_len].unsqueeze(0)  # (1, current_len)
                tok_logits, _, _ = transformer_model(z_seq, a_prefix)  # (1, L, K)
                next_logits = tok_logits[:, -1, :]           # (1, K)
                next_z = F.one_hot(next_logits.argmax(-1), num_classes=num_categories).float()  # (1, K)
                frame = catvae_model.decoder(next_z)               # (1, 3, 64, 64)
                trans_preds.append(frame)
                z_seq = torch.cat([z_seq, next_z.unsqueeze(1)], dim=1)  # (1, L+1, K)

            # 真实帧
            for h in horizons:
                if h <= rollout_len and h < eval_obs.shape[1]:
                    gt = eval_obs[traj_i, h:h+1]             # (1, 3, 64, 64)
                    p_rssm  = psnr(rssm_preds[h-1],  gt).item()
                    p_trans = psnr(trans_preds[h-1], gt).item()
                    psnr_rssm_all[h].append(p_rssm)
                    psnr_trans_all[h].append(p_trans)

    psnr_rssm_mean  = [np.mean(psnr_rssm_all[h])  for h in horizons]
    psnr_trans_mean = [np.mean(psnr_trans_all[h]) for h in horizons]

    print('各预测步数的 PSNR（dB）：')
    print(f'{"预测步数":>10}  {"RSSM":>10}  {"Transformer":>12}')
    for h, r, t in zip(horizons, psnr_rssm_mean, psnr_trans_mean):
        print(f'{h:>10}  {r:>10.2f}  {t:>12.2f}')



if __name__ == '__main__':
    parameter = Parameter()
    rssm_path = r"C:\Users\Lenovo\Desktop\MyGithub\WorldModel\WorldModel\RSSM\rssm.pt"
    load_rssm(rssm_path=rssm_path, latent_dim=parameter.latent_dim, action_dim=parameter.action_dim, hidden_dim=parameter.hidden_dim)