import numpy as np
import torch
import matplotlib.pyplot as plt
from config import _configure_cjk_font
from parameters import Parameter

# 合成图形图像数据集。
def make_shape_images(n=1000, size=64, seed=0):
    rng = np.random.RandomState(seed)
    imgs = np.zeros((n, 3, size, size), dtype=np.float32)
    for i in range(n):
        # 背景
        bg = rng.uniform(0.05, 0.2, (3, 1, 1)).astype(np.float32)
        imgs[i] = bg
        # 随机形状：圆形或矩形
        color = rng.uniform(0.4, 1.0, 3).astype(np.float32)
        cx = rng.randint(10, size - 10)
        cy = rng.randint(10, size - 10)
        r = rng.randint(5, 14)
        shape_type = rng.randint(0, 2)
        for c in range(3):
            if shape_type == 0:  # 圆形
                for y in range(size):
                    for x in range(size):
                        if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                            imgs[i, c, y, x] = color[c]
            else:  # 矩形
                x0, x1 = max(0, cx - r), min(size, cx + r)
                y0, y1 = max(0, cy - r), min(size, cy + r)
                imgs[i, c, y0:y1, x0:x1] = color[c]
    return torch.from_numpy(imgs)

if __name__ == "__main__":
    _CJK_FONT = _configure_cjk_font()
    parameter = Parameter()
    print(f'正在生成{parameter.n_img}张合成图形图像...')
    images = make_shape_images(n=parameter.n_img, size=parameter.img_size, seed=parameter.seed)
    print('图像张量形状：', images.shape, '  dtype:', images.dtype)
    # 快速健全性检查
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.axis('off')
        ax.set_title(f'图像 {i}')
    plt.suptitle('合成图像样例', y=1.02)
    plt.tight_layout()
    plt.show()