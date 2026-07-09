import argparse, smart_logger
from smart_logger.parameter.ParameterTemplate import ParameterTemplate


class Parameter(ParameterTemplate):
    
    def __init__(self, config_path=None, debug=False):
        super(Parameter, self).__init__(config_path, debug)

    def parse(self):
        parser = argparse.ArgumentParser(description=smart_logger.experiment_config.EXPERIMENT_TARGET)
        self.img_size = 64
        parser.add_argument('--img_size', type=int, default=self.img_size, metavar='N', help='image size')
        
        self.img_ch = 3
        parser.add_argument('--img_ch', type=int, default=self.img_ch, metavar='N', help="channels of img")

        self.latent_dim = 32
        parser.add_argument('--latent_dim', type=int, default=self.latent_dim, metavar='N', help='potential variable of VAE')

        self.hidden_dim = 128
        parser.add_argument('--hidden_dim', type=int, default=self.hidden_dim, metavar='N', help='hidden space of RSSM')

        self.action_dim = 2
        parser.add_argument('--action_dim', type=int, default=self.action_dim, metavar='N', help='action space dim')

        self.actor_critic_hidden = 128
        parser.add_argument('--actor_critic_hidden', type=int, default=self.actor_critic_hidden, metavar='N', help='actor-critic hidden dim')

        self.episode_len = 20
        parser.add_argument('--episode_len', type=int, default=self.episode_len, metavar='N', help='每个回合的步数')

        self.n_iter = 30
        parser.add_argument('--n_iter', type=int, default=self.n_iter, metavar='N', help='外层训练迭代次数')

        self.n_eval = 10
        parser.add_argument('--n_eval', type=int, default=self.n_eval, metavar='N', help='eval')

        self.batch_size = 4
        parser.add_argument('--batch_size', type=int, default=self.batch_size, metavar='N', help='batch size')

        self.imagine_h = 10
        parser.add_argument('--imagine_h', type=int, default=self.imagine_h, metavar='N', help='想象时域')

        self.lam_gae = 0.95
        parser.add_argument('--lam_gae', type=float, default=self.lam_gae, metavar='N', help='gae')

        self.gamma = 0.99
        parser.add_argument('--gamma', type=float, default=self.gamma, metavar='N', help='discount factor in RL')

        self.lr_wr = 3e-4
        parser.add_argument('--lr_wr', type=float, default=self.lr_wr, metavar='N', help='learing rate of world model')

        self.lr_ac = 3e-4 
        parser.add_argument('--lr_ac', type=float, default=self.lr_ac, metavar='N', help='learning rate of actor critic')

        return parser.parse_args()

    
if __name__ == '__main__':

    def main():
        parameter = Parameter()
        print(parameter)
    main()
