import torch
import torch.nn as nn
from config import DEVICE
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from VAE.model import Encoder, Decoder


def _load_vae_checkpoint(path, encoder, decoder):
    """
    加载VAE模型 
    """
    ckpt = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        enc_state = {
            k.removeprefix('encoder.'): v
            for k, v in state.items()
            if k.startswith('encoder.')
        }
        dec_state = {
            k.removeprefix('decoder.'): v
            for k, v in state.items()
            if k.startswith('decoder.')
        }
        encoder.load_state_dict(enc_state)
        decoder.load_state_dict(dec_state)
        return True
    if 'encoder' in ckpt and 'decoder' in ckpt:
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        return True
    raise KeyError(f'未识别的权重文件格式: {list(ckpt.keys())[:10]}')


if __name__ == '__main__':

    img_ch = 3
    latent_dim = 32
    hidden_dim = 128
    encoder = Encoder(img_ch=img_ch, latent_dim=latent_dim)
    decoder = Decoder(img_ch=img_ch, latent_dim=latent_dim, hidden_dim=hidden_dim)
    ckpt_path = Path(r'C:\Users\Lenovo\Desktop\MyGithub\WorldModel\WorldModel\VAE\vae_encoder.pt')
    try:
        _load_vae_checkpoint(ckpt_path, encoder, decoder)
        print(f'已从 {ckpt_path} 加载 VAE 权重')
    except Exception as e:
        print(f'无法从 {ckpt_path} 加载 VAE 权重文件（{e}），使用随机初始化。')

    VAE_CHECKPOINT_PATH = ckpt_path
    encoder.eval()
    decoder.eval()
    for p in list(encoder.parameters()) + list(decoder.parameters()):
        p.requires_grad_(False)

    print(f'编码器参数量: {sum(p.numel() for p in encoder.parameters()):,}')
    print(f'解码器参数量: {sum(p.numel() for p in decoder.parameters()):,}')