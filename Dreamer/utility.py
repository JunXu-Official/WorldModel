import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from collections import deque


# 让 Colab 和新环境优先使用支持中文的字体，避免标题和坐标轴显示成方框。
def _configure_cjk_font():
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
    ]
    for family in preferred:
        try:
            font_manager.findfont(family, fallback_to_default=False)
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [family] + [f for f in mpl.rcParams.get("font.sans-serif", []) if f != family]
            mpl.rcParams["axes.unicode_minus"] = False
            return family
        except Exception:
            pass

    font_path = Path.home() / ".cache" / "notebook-fonts" / "NotoSansCJKsc-Regular.otf"
    if not font_path.exists():
        try:
            import urllib.request
            font_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            font_path = None

    if font_path and font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [family] + [f for f in preferred if f != family]
        mpl.rcParams["axes.unicode_minus"] = False
        return family

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    return None

_CJK_FONT = _configure_cjk_font()
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def _resolve_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE = _resolve_device()
USE_TPU = DEVICE.type == 'xla'
USE_CUDA = DEVICE.type == 'cuda'
LOAD_DEVICE = torch.device('cpu') if USE_TPU else DEVICE


def optimizer_step(optimizer, scaler=None):
    if USE_TPU:
        xm.optimizer_step(optimizer)
    elif scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def init_buffer():
    """
    """
    # 回放缓冲区存储轨迹字典。
    replay_buffer = deque(maxlen=200)
    return replay_buffer

def init_optimizer(models, lr_ac, lr_wr):
    """
    """
    # 世界模型优化器涵盖编码器、解码器和 RSSM
    wm_params = []
    for k, v in models.items():
        if k != 'actor' and k != 'critic':
            wm_params += list(v.parameters())
        
        if k == 'actor':
            opt_actor = torch.optim.Adam(models['actor'].parameters(),  lr=lr_ac)
        if k == 'critic':
            opt_critic = torch.optim.Adam(models['critic'].parameters(), lr=lr_ac)
    
    opt_wm = torch.optim.Adam(wm_params, lr=lr_wr)
    
    opts = {
        "opt_wm": opt_wm,
        "opt_actor": opt_actor,
        "opt_critic": opt_critic
    }
    
    return wm_params, opts


def action_to_onehot(action_int, action_dim):
    """标量整数 -> (1, action_dim) one-hot 张量。"""
    oh = torch.zeros(1, action_dim, device=DEVICE)
    oh[0, action_int] = 1.0
    return oh    

def obs_to_tensor(obs):
    """将 HWC numpy float32 转换为 (1, 3, H, W) 张量。"""
    t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
    return t.to(DEVICE)