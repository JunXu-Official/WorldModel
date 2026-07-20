import argparse, smart_logger
from smart_logger.parameter.ParameterTemplate import ParameterTemplate


class Parameter(ParameterTemplate):
    def __init__(self, config_path=None, debug=False):
        super(Parameter, self).__init__(config_path, debug)

    def parse(self):
        parser = argparse.ArgumentParser(description=smart_logger.experiment_config.EXPERIMENT_TARGET)
        
        self.num_categories = 32
        parser.add_argument('--num_categories', type=int, default=self.num_categories, metavar='N', help='离散词表大小')

        self.z_dim = 32
        parser.add_argument('--z_dim', type=int, default=self.z_dim, metavar='N', help='每个token的嵌入维度')

        self.tau = 1.0
        parser.add_argument('--tau', type=float, default=self.tau, metavar='N', help='tau')

        self.img_size = 64
        parser.add_argument('--img_size', type=int, default=self.img_size, metavar='N', help='图片尺寸')

        self.n_img = 1000
        parser.add_argument('--n_img', type=int, default=self.n_img, metavar='N', help='生成图片的数量')

        self.seed = 0
        parser.add_argument('--seed', type=int, default=self.seed, metavar='N', help='随机种子')

        self.batch_size = 64
        parser.add_argument('--batch_size', type=int, default=self.batch_size, metavar='N', help='批处理大小')

        self.catVAE_epoches = 200
        parser.add_argument('--catVAE_epoches', type=int, default=self.catVAE_epoches, metavar='N', help='catVAE的训练迭代次数')

        self.d_model = 128
        parser.add_argument('--d_model', type=int, default=self.d_model, metavar='N', help='transformer的隐空间')

        self.n_heads = 4
        parser.add_argument('--n_heads', type=int, default=self.n_heads, metavar='N', help='Transformer的头数')

        self.n_layers = 2
        parser.add_argument('--n_layers', type=int, default=self.n_layers, metavar='N', help='Transformer层数')

        self.seq_len = 20
        parser.add_argument('--seq_len', type=int, default=self.seq_len, metavar='N', help='Transformer的轨迹长度')

        self.n_traj = 200
        parser.add_argument('--n_traj', type=int, default=self.n_traj, metavar='N', help='轨迹数量')

        self.n_actions = 2
        parser.add_argument('--n_actions', type=int, default=self.n_actions, metavar='N', help='动作维度')

        self.transformer_epoches = 100
        parser.add_argument('--transformer_epoches', type=int, default=self.transformer_epoches, metavar='N', help='Transforer的训练迭代次数')

        self.transformer_batch_size = 32
        parser.add_argument('--transformer_batch_size', type=int, default=self.transformer_batch_size, metavar='N', help='Transformer训练的批处理大小')

        self.transformer_lr = 1e-3
        parser.add_argument('--transformer_lr', type=float, default=self.transformer_lr, metavar='N', help='Transformer训练的学习率')

        self.latent_dim = 32
        parser.add_argument('--latent_dim', type=int, default=self.latent_dim, metavar='N', help='')

        self.action_dim = 1
        parser.add_argument('--action_dim', type=int, default=self.action_dim, metavar='N', help='')

        self.hidden_dim = 128
        parser.add_argument('--hidden_dim', type=int, default=self.hidden_dim, metavar='N', help='hidden dim')

        self.rollout_len = 20
        parser.add_argument('--rollout_len', type=int, default=self.rollout_len, metavar='N', help='评估轨迹长度')
        
        self.n_eval = 5
        parser.add_argument('--n_eval', type=int, default=self.n_eval, metavar='N', help='评估次数')






        return parser
    

if __name__ == "__main__":
    def main():
        parameter = Parameter()
        print(parameter)
    
    main()