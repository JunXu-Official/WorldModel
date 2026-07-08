import numpy as np
from parameters import Parameter


class SyntheticEnv:
    """带图像观测的简单合成控制环境。"""
    def __init__(self, parameter, seed=None):
        self.parameter = parameter
        self.episode_len = self.parameter.episode_len
        self.img_size = self.parameter.img_size
        self.rng = np.random.default_rng(seed)
        self.pos = 0.0
        self.step_count = 0

    def _render(self):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        bar_x = int((self.pos + 1.0) / 2.0 * (self.img_size - 1))
        bar_x = np.clip(bar_x, 0, self.img_size - 1)
        img[:, max(0, bar_x - 2): bar_x + 3, 0] = 1.0  # 红色通道
        # 加入轻微背景噪声，使编码器面临非平凡任务
        img += self.rng.uniform(0, 0.05, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)

    def reset(self):
        self.pos = float(self.rng.uniform(-0.8, 0.8))
        self.step_count = 0
        return self._render()

    def step(self, action):
        """action: int（0 或 1）。返回 (obs, reward, done)。"""
        prev_abs = abs(self.pos)
        delta = 0.1 if action == 1 else -0.1
        self.pos = float(np.clip(self.pos + delta, -1.0, 1.0))
        reward = 1.0 if abs(self.pos) < prev_abs else -1.0
        self.step_count += 1
        done = self.step_count >= self.episode_len
        return self._render(), reward, done

if __name__ == '__main__':
    # 快速完整性检查
    parameter = Parameter()
    env = SyntheticEnv(parameter=parameter, seed=0)
    obs = env.reset()
    print(f'观测形状: {obs.shape}, 数据类型: {obs.dtype}, 范围: [{obs.min():.2f}, {obs.max():.2f}]')
    obs2, r, done = env.step(1)
    print(f'执行动作后: 奖励={r}, 回合结束={done}')