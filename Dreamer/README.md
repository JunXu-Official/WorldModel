# Dreamer 实现

基于 Dreamer 算法的世界模型实现，用于在合成环境中学习策略。

## 项目结构

```
Dreamer/
├── main.py              # 主训练脚本
├── env.py               # 合成环境实现
├── rssm.py              # Recurrent State-Space Model (RSSM)
├── actor_critic.py      # Actor-Critic 算法
├── imagine_rollout.py   # 想象推演
├── wm_update.py         # 世界模型更新
├── collect_data.py      # 数据收集
├── load_model.py        # 模型加载和初始化
├── parameters.py        # 参数配置
├── utility.py           # 工具函数
├── buffer.py            # 经验回放缓冲区
└── dreamer.pt           # 保存的模型权重
```

## 核心算法

Dreamer 算法包含三个主要组件：

1. **世界模型 (World Model)**
   - 编码器 (Encoder): 将观测压缩到潜在空间
   - 解码器 (Decoder): 从潜在状态重建观测
   - RSSM: 建模状态转移 dynamics

2. **行为学习 (Behavior Learning)**
   - Actor: 从潜在状态预测动作
   - Critic: 评估状态价值
   - 通过想象推演学习策略

3. **想象推演 (Imagination Rollout)**
   - 在潜在空间中进行多步预测
   - 使用 GAE 计算优势函数
   - 更新 Actor 和 Critic

## 运行方法

```bash
# 训练模型
python main.py

# 使用特定配置
python main.py --config config.yaml
```

## 参数配置

主要参数在 `parameters.py` 中定义：

- `latent_dim`: 潜在空间维度
- `hidden_dim`: RSSM 隐藏层维度
- `action_dim`: 动作空间维度
- `ac_hidden`: Actor-Critic 隐藏层维度
- `episode_len`: 回合长度
- `n_iter`: 训练迭代次数
- `n_eval`: 评估回合数
- `batch_size`: 批次大小
- `imagine_h`: 想象推演步数
- `gamma`: 折扣因子
- `lr_wr`: 世界模型学习率
- `lr_ac`: Actor-Critic 学习率

## 输出

训练完成后会生成：

- `dreamer.pt`: 模型权重文件
- 训练曲线图（回合奖励、重建损失、KL 散度等）
- 评估结果（真实奖励 vs 预测奖励的相关性）

## 依赖

- PyTorch
- NumPy
- Matplotlib
- smart_logger
