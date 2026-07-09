import torch
from utility import DEVICE, action_to_onehot, obs_to_tensor
import torch.nn.functional as F
import torch.nn as nn


def world_model_update(models, batch, action_dim, wm_params, opt_wm):
    """
    """
    encoder = models['encoder']
    decoder = models['decoder']
    rssm = models['rssm']
    reward_model = models['reward_model']
    encoder.train()
    decoder.train()
    rssm.train()
    opt_wm.zero_grad()
    total_recon = 0.0
    total_kl = 0.0
    total_reward = 0.0
    count = 0

    for traj in batch:
        obs_list = traj['obs']     # T+1 个 numpy 数组列表 (H,W,3)
        act_list = traj['actions'] # T 个整数列表
        T = len(act_list)
        h, z = rssm.initial_state(1)
        for t in range(T):
            obs_t = obs_to_tensor(obs_list[t])           # (1,3,64,64)
            obs_next = obs_to_tensor(obs_list[t + 1])
            a_oh = action_to_onehot(act_list[t], action_dim=action_dim)        # (1, action_dim)
            # 编码当前观测
            enc_z, enc_mu, enc_logvar = encoder.encode(obs_t)
            # 由编码器嵌入得到后验
            z_post, post_mu, post_std = rssm.posterior(h, enc_z)
            # 由确定性状态得到先验
            _, prior_mu, prior_std = rssm.prior(h)
            # 重建下一帧观测
            h_next = rssm.step(h, z_post.detach(), a_oh)
            recon  = decoder(z_post, h_next)
            # 从潜在状态预测实际转移奖励
            reward_target = torch.tensor([traj['rewards'][t]], device=DEVICE, dtype=torch.float32)
            reward_pred   = reward_model(h, z_post, a_oh)
            # 重建损失（每像素 MSE，在空间维度上取均值）
            recon_loss = F.mse_loss(recon, obs_next, reduction='mean')
            # KL 散度：后验 || 先验（封闭形式，均为高斯分布）
            kl = 0.5 * (
                (post_std / prior_std).pow(2)
                + ((post_mu - prior_mu) / prior_std).pow(2)
                - 1
                + 2 * prior_std.log()
                - 2 * post_std.log()
            ).sum(dim=-1).mean()
            reward_loss = F.mse_loss(reward_pred, reward_target)
            total_recon = total_recon + recon_loss
            total_kl = total_kl    + kl
            total_reward = total_reward + reward_loss
            count += 1
            # 推进状态（detach 以避免跨步的时间反向传播）
            h = h_next.detach()
            z = z_post.detach()
    loss = (total_recon + total_kl + total_reward) / max(count, 1)
    loss.backward()
    nn.utils.clip_grad_norm_(wm_params, 100.0)
    opt_wm.step()
    return (total_recon / count).item(), (total_kl / count).item(), (total_reward / count).item()
    

