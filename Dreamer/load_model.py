import torch
from config import DEVICE
from parameters import Parameter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from VAE.model import Encoder, Decoder
from rssm import RSSM
from ac import Actor, Critic, RewardModel

def obs_to_tensor(obs):
    """
    """
    t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
    return t.to(DEVICE)

def init_model(parameter):
    """
    """
    encoder = Encoder(img_ch=parameter.img_ch, latent_dim=parameter.latent_dim).to(DEVICE)
    decoder = Decoder(img_ch=parameter.img_ch, latent_dim=parameter.latent_dim, hidden_dim=parameter.hidden_dim).to(DEVICE)
    rssm = RSSM(latent_dim=parameter.latent_dim, hidden_dim=parameter.hidden_dim, action_dim=parameter.action_dim).to(DEVICE)
    actor = Actor(latent_dim=parameter.latent_dim, hidden_dim=parameter.hidden_dim, action_dim=parameter.action_dim, ac_hidden=parameter.actor_critic_hidden).to(DEVICE)
    critic = Critic(latent_dim=parameter.latent_dim, hidden_dim=parameter.hidden_dim, ac_hidden=parameter.actor_critic_hidden).to(DEVICE)
    reward_model = RewardModel(latent_dim=parameter.latent_dim, hidden_dim=parameter.hidden_dim, action_dim=parameter.action_dim, ac_hidden=parameter.actor_critic_hidden).to(DEVICE)
    models = {
        "encoder": encoder,
        "decoder": decoder,
        "rssm": rssm,
        "actor": actor,
        "critic": critic,
        "rm": reward_model
    }
    return models

# 尝试从前序项目加载权重。
def _load_encoder_decoder_from_vae_checkpoint(path, encoder):
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    if isinstance(ckpt, dict) and 'encoder' in ckpt:
        enc_state = {k.replace('fc_log_var', 'fc_logvar'): v for k, v in ckpt['encoder'].items()}
        encoder.load_state_dict(enc_state, strict=True)
        return '仅编码器'

    enc_state = {}
    dec_state = {}
    for key, value in state.items():
        if key.startswith('encoder.'):
            enc_key = key[len('encoder.'):].replace('fc_log_var', 'fc_logvar')
            enc_state[enc_key] = value
        elif key.startswith('decoder.'):
            dec_state[key[len('decoder.'):]] = value

    if not enc_state:
        raise KeyError(f'无法识别的 VAE 权重文件格式: {list(ckpt.keys())[:10] if isinstance(ckpt, dict) else type(ckpt)}')

    encoder.load_state_dict(enc_state, strict=True)
    return '仅编码器来自 model_state_dict'


def _load_rssm_from_rssm_checkpoint(encoder_path, rssm_path, models):
    """
    """
    vae_ckpt_candidates = [Path(encoder_path), Path('notebooks') / encoder_path]
    vae_ckpt_path = next((p for p in vae_ckpt_candidates if p.exists()), None)
    if vae_ckpt_path is not None:
        try:
            vae_ckpt_format = _load_encoder_decoder_from_vae_checkpoint(vae_ckpt_path, encoder=models["encoder"])
            print(f'已从 {vae_ckpt_path} 加载编码器/解码器权重（{vae_ckpt_format}）')
        except Exception as e:
            print(f'无法从 {vae_ckpt_path} 加载编码器/解码器（{e}），使用随机初始化')
    else:
        print('未找到 vae_encoder.pt，使用随机初始化编码器')

    rssm_path = next((p for p in [Path(rssm_path), Path('notebooks') / rssm_path] if p.exists()), None)
    if rssm_path is not None:
        try:
            state = torch.load(rssm_path, map_location=DEVICE)
            if isinstance(state, dict) and 'rssm_state_dict' in state:
                sd = state['rssm_state_dict']
                models["rssm"].load_state_dict(sd, strict=True)
                print(
                    f"已从 {rssm_path} 加载 RSSM 权重 "
      
                )
            elif isinstance(state, dict) and 'rssm' in state:
                models["rssm"].load_state_dict(state['rssm'], strict=False)
                print(f'已从 {rssm_path} 加载 RSSM 权重（旧版 rssm 键）')
            else:
                models["rssm"].load_state_dict(state, strict=False)
                print(f'已从 {rssm_path} 加载 RSSM 权重（原始 state_dict）')
        except Exception as e:
            print(f'无法加载 RSSM（{e}），使用随机初始化')
    else:
        print('未找到 rssm.pt，使用随机初始化 RSSM')
    print('\n各模型参数量:')
    for k, v in models.items():
        n = sum(p.numel() for p in v.parameters())
        print(f"{k}:{n}")

if __name__ == "__main__":

    parameter = Parameter()
    models = init_model(parameter=parameter)
    encoder_path = r"C:\Users\Lenovo\Desktop\MyGithub\WorldModel\WorldModel\VAE\vae_encoder.pt"
    rssm_path = r"C:\Users\Lenovo\Desktop\MyGithub\WorldModel\WorldModel\RSSM\rssm.pt"
    _load_rssm_from_rssm_checkpoint(
        encoder_path=encoder_path,
        rssm_path=rssm_path,
        models=models
    )
