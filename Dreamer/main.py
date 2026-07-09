from collect_data import collect_episode, get_rssm_states_from_traj, supervised_policy_update
from utility import init_buffer, init_optimizer
import random
from wm_update import world_model_update
from imagine_rollout import behavior_update
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



def train(models, n_iter, batch_size, img_h, img_size, episode_len, action_dim, wm_params, opts):
    """
    """
    # 指标历史记录。
    ep_rewards      = []
    recon_losses    = []
    kl_losses       = []
    reward_losses   = []
    policy_losses   = []
    actor_entropies = []

    for iteration in range(n_iter):
        # --- 收集一个回合 ---
        traj, ep_reward = collect_episode(
            models=models,
            img_size=img_size,
            episode_len=episode_len,
            action_dim=action_dim,
            env_seed=iteration, 
            deterministic=False, 
            epsilon=0.10)
        replay_buffer = init_buffer()
        replay_buffer.append(traj)
        ep_rewards.append(ep_reward)
        # --- 世界模型更新 ---
        buf_list = list(replay_buffer)
        n_sample = min(batch_size, len(buf_list))
        batch = random.sample(buf_list, n_sample)
        recon_l, kl_l, reward_l = world_model_update(
            models=models, 
            batch=batch,
            action_dim=action_dim,
            wm_params=wm_params,
            opt_wm=opts['opt_wm'])
        recon_losses.append(recon_l)
        kl_losses.append(kl_l)
        reward_losses.append(reward_l)
        # --- Critic 更新（想象推演）---
        h_states, z_states = get_rssm_states_from_traj(
            models=models,
            action_dim=action_dim,
            traj=traj)
        entropy = behavior_update(
            models=models,
            opts=opts,
            action_dim=action_dim,
            start_h=h_states,
            start_z=z_states,
            horizon=img_h)
        actor_entropies.append(entropy)

        # --- 从回放中使用简单专家策略更新 Actor ---
        policy_l = supervised_policy_update(
            models=models,
            opts=opts,
            batch=batch,
            action_dim=action_dim)
        policy_losses.append(policy_l)

        if (iteration + 1) % 10 == 0:
            print(
                f'迭代 {iteration+1:3d} | '
                f'回合奖励={ep_reward:+.1f} | '
                f'重建={recon_l:.4f} | '
                f'KL={kl_l:.4f} | '
                f'奖励={reward_l:.4f} | '
                f'策略={policy_l:.4f} | '
                f'Actor熵={entropy:.4f}'
            )
    return ep_rewards


def imagined_rollout_rewards(models, action_dim, start_h, start_z, horizon=10, deterministic=True):
    """使用 Actor 和奖励模型在想象空间中前向推演。"""
    models['actor'].eval()
    models['reward_model'].eval()
    models['rssm'].eval()
    h, z = start_h.clone(), start_z.clone()
    im_rewards = []
    im_entropies = []

    with torch.no_grad():
        for _ in range(horizon):
            logits = models['actor'](h, z)
            dist   = torch.distributions.Categorical(logits=logits)
            im_entropies.append(dist.entropy().mean().item())
            if deterministic:
                a = torch.argmax(logits, dim=-1)
            else:
                a = dist.sample()
            a_oh   = F.one_hot(a, num_classes=action_dim).float()
            r_hat  = models['reward_model'](h, z, a_oh).mean().item()
            im_rewards.append(r_hat)
            h = models['rssm'].step(h, z, a_oh)
            prior_out = models['rssm'].prior_net(h)
            z, _ = prior_out.chunk(2, dim=-1)

    return im_rewards, im_entropies


def rollout(models, n_eval, episode_len, img_size, action_dim, ep_rewards):
    real_reward_sums  = []
    imag_reward_sums  = []
    imag_entropies_ev = []
    models['encoder'].eval()
    models['rssm'].eval()
    models['actor'].eval()
    models['reward_model'].eval()
    for ep_i in range(n_eval):
        # 收集真实回合
        traj, ep_r = collect_episode(
            models=models, 
            img_size=img_size,
            episode_len=episode_len,
            action_dim=action_dim,
            env_seed=1000 + ep_i,
            deterministic=True)
        real_reward_sums.append(sum(traj['rewards']))
        # 以第一步的 RSSM 状态作为想象推演的起点
        h0, z0 = get_rssm_states_from_traj(
            models=models,
            action_dim=action_dim,
            traj=traj)
        seed_h  = h0[0:1]   # 单步
        seed_z  = z0[0:1]
        # 想象奖励与熵
        im_r, ents = imagined_rollout_rewards(
            models=models,
            action_dim=action_dim,
            start_h=seed_h,
            start_z=seed_z,
            horizon=episode_len,
            deterministic=True)
        imag_reward_sums.append(sum(im_r))
        imag_entropies_ev.append(float(np.mean(ents)))
    print(f'在 {n_eval} 个回合上的评估完成。')
    print(f'真实奖励均值（完整回合）:{np.mean(real_reward_sums):.3f}')
    print(f'预测奖励均值（想象推演）:{np.mean(imag_reward_sums):.3f}')
    print(f'想象轨迹熵均值:{np.mean(imag_entropies_ev):.4f}')
    # 计算想象奖励与真实奖励之和的 Pearson 相关系数
    r_real = np.array(real_reward_sums)
    r_imag = np.array(imag_reward_sums)
    if r_real.std() > 1e-8 and r_imag.std() > 1e-8:
        rho = np.corrcoef(r_real, r_imag)[0, 1]
    else:
        rho = 0.0

    print(f'奖励相关性 rho（预测值与真实值，{episode_len} 步）: {rho:.4f}')
    print(f'想象轨迹熵均值:{np.mean(imag_entropies_ev):.4f}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Dreamer 自评估', fontsize=13)
    # 训练过程中的回合奖励
    axes[0].plot(ep_rewards, color='steelblue', label='训练')
    axes[0].set_title('回合奖励（训练阶段）')
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('总奖励')
    axes[0].legend()

    # 奖励相关性散点图
    axes[1].scatter(r_real, r_imag, color='tomato', alpha=0.7)
    axes[1].set_title(f'奖励相关性（rho={rho:.3f}）')
    axes[1].set_xlabel(f'真实奖励（完整回合）')
    axes[1].set_ylabel('预测奖励（想象推演）')
    lims = [min(r_real.min(), r_imag.min()) - 0.5, max(r_real.max(), r_imag.max()) + 0.5]
    axes[1].plot(lims, lims, 'k--', alpha=0.3, label='理想线')
    axes[1].legend()

    # 各评估回合的想象轨迹熵
    axes[2].bar(range(n_eval), imag_entropies_ev, color='mediumpurple', alpha=0.8)
    axes[2].set_title('想象轨迹熵')
    axes[2].set_xlabel('评估回合')
    axes[2].set_ylabel('平均熵（奈特）')
        
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def save_pt(save_path, models, latent_dim, action_dim, hidden_dim, ac_hidden_dim):
    checkpoint = {
    'encoder': models['encoder'].state_dict(),
    'decoder': models['decoder'].state_dict(),
    'rssm': models['rssm'].state_dict(),
    'actor': models['actor'].state_dict(),
    'critic': models['critic'].state_dict(),
    'reward_model': models['reward_model'].state_dict(),
    'hyperparams': {
        'latent_dim':  latent_dim,
        'hidden_dim':  hidden_dim,
        'action_dim':  action_dim,
        'ac_hidden':   ac_hidden_dim,
    },
}
    torch.save(checkpoint, save_path)
    print(f'权重文件已保存至 {save_path}')


if __name__ == '__main__':
    from parameters import Parameter
    from load_model import init_model
    parameter = Parameter()
    models = init_model(
        img_ch=parameter.img_ch,
        latent_dim=parameter.latent_dim,
        action_dim=parameter.action_dim,
        hidden_dim=parameter.hidden_dim,
        ac_hidden_dim=parameter.actor_critic_hidden
    )
    wm_params, opts = init_optimizer(
        models=models,
        lr_ac=parameter.lr_ac,
        lr_wr=parameter.lr_wr
    )
    # train
    ep_rewards = train(
        models=models,
        n_iter=parameter.n_iter,
        batch_size=parameter.batch_size,
        img_h=parameter.imagine_h,
        img_size=parameter.img_size,
        episode_len=parameter.episode_len,
        action_dim=parameter.action_dim,
        wm_params=wm_params,
        opts=opts
    )

    rollout(
        models=models,
        n_eval=parameter.n_eval,
        episode_len=parameter.episode_len,
        img_size=parameter.img_size,
        action_dim=parameter.action_dim,
        ep_rewards=ep_rewards
    )

    save_pt(
        save_path=r"C:\Users\Lenovo\Desktop\MyGithub\WorldModel\WorldModel\Dreamer\dreamer.pt",
        models=models,
        latent_dim=parameter.latent_dim,
        action_dim=parameter.action_dim,
        hidden_dim=parameter.hidden_dim,
        ac_hidden_dim=parameter.actor_critic_hidden
    )
