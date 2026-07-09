import torch
import torch.nn as nn 
from env import SyntheticEnv
from utility import obs_to_tensor, action_to_onehot, DEVICE
import numpy as np
import random
import torch.nn.functional as F

def collect_episode(models, img_size, episode_len, action_dim, env_seed=None, deterministic=False, epsilon=0.05):
    """使用当前 Actor 收集一个回合的数据。"""
    env = SyntheticEnv(img_size=img_size, episode_len=episode_len, seed=env_seed)
    obs = env.reset()
    traj = {'obs': [obs], 'actions': [], 'rewards': []}
    h, z = models['rssm'].initial_state(1)
    models['actor'].eval()
    models['encoder'].eval()
    models['rssm'].eval()
    total_reward = 0.0
    done = False
    with torch.no_grad():
        while not done:
            obs_t = obs_to_tensor(obs)
            enc_z, _, _ = models['encoder'].encode(obs_t)
            z_post, _, _ = models['rssm'].posterior(h, enc_z)
            bar_pos = obs_to_bar_pos(obs)
            logits = models['actor'](h, z_post, bar_pos=bar_pos)
            dist   = torch.distributions.Categorical(logits=logits)
            if deterministic:
                a_int = int(torch.argmax(logits, dim=-1).item())
            else:
                if random.random() < epsilon:
                    a_int = random.randint(0, action_dim - 1)
                else:
                    a_int = int(dist.sample().item())

            obs_next, reward, done = env.step(a_int)
            a_oh = action_to_onehot(a_int, action_dim)
            h = models['rssm'].step(h, z_post, a_oh)
            z = z_post

            traj['obs'].append(obs_next)
            traj['actions'].append(a_int)
            traj['rewards'].append(reward)
            obs = obs_next
            total_reward += reward

    return traj, total_reward


def obs_to_bar_pos(obs):
    """从红色通道估计滑块位置，返回归一化标量。"""
    red_profile = obs[:, :, 0].mean(axis=0)
    bar_x = int(np.argmax(red_profile))
    denom = max(obs.shape[1] - 1, 1)
    bar_pos = (2.0 * bar_x / denom) - 1.0
    return torch.tensor([[bar_pos]], device=DEVICE, dtype=torch.float32)


def get_rssm_states_from_traj(models, action_dim,  traj):
    """在轨迹上重新运行 RSSM，获取后验 (h, z) 对，用于想象推演。"""
    models['encoder'].eval()
    models['rssm'].eval()
    h, z = models['rssm'].initial_state(1)
    h_list, z_list = [], []
    with torch.no_grad():
        for t, a_int in enumerate(traj['actions']):
            obs_t = obs_to_tensor(traj['obs'][t])
            enc_z, _, _ = models['encoder'].encode(obs_t)
            z_post, _, _ = models['rssm'].posterior(h, enc_z)
            h_list.append(h)
            z_list.append(z_post)
            a_oh = action_to_onehot(a_int, action_dim=action_dim)
            h = models['rssm'].step(h, z_post, a_oh)
    return torch.cat(h_list, dim=0), torch.cat(z_list, dim=0)  # (T, dim)


def expert_action_from_obs(obs):
    """为合成滑块任务返回向中心移动的专家动作。"""
    red_profile = obs[:, :, 0].mean(axis=0)
    bar_x = int(np.argmax(red_profile))
    center = obs.shape[1] // 2
    return 1 if bar_x < center else 0


def supervised_policy_update(models, opts, batch, action_dim):
    """训练 Actor 模仿回放轨迹上向中心移动的专家策略。"""
    models['actor'].train()
    losses = []
    for traj in batch:
        h_states, z_states = get_rssm_states_from_traj(
            models=models,
            action_dim=action_dim,
            traj=traj
        )
        targets = torch.tensor([expert_action_from_obs(obs) for obs in traj['obs'][:-1]], device=DEVICE, dtype=torch.long)
        bar_pos = torch.cat([obs_to_bar_pos(obs) for obs in traj['obs'][:-1]], dim=0)
        logits = models['actor'](h_states, z_states, bar_pos=bar_pos)
        losses.append(F.cross_entropy(logits, targets))

    loss = torch.stack(losses).mean()
    opts['opt_actor'].zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(models['actor'].parameters(), 100.0)
    opts['opt_actor'].step()
    return loss.item()