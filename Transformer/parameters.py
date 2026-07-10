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

        self.epoches = 200
        parser.add_argument('--epoches', type=int, default=self.epoches, metavar='N', help='训练迭代次数')

        return parser
    

if __name__ == "__main__":
    def main():
        parameter = Parameter()
        print(parameter)
    
    main()