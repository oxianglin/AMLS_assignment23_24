import os
import sys
import argparse

sys.path.insert(0, f'{os.getcwd()}/A')
import pneumoniamnist as pnm

sys.path.insert(0, f'{os.getcwd()}/B')
from pathmnist import Pathmnist

class Amls():
    def __init__(self, args:argparse.Namespace=()):
        super().__init__()
        self.mnist = args.mnist
        self.nettype = args.nettype
        if (self.nettype > 1): #To change based on the number of nettype supported
            self.nettype = 1
        self.lib = args.lib
        self.debug = args.debug

    @classmethod
    def argparser(cls):
        parser = argparse.ArgumentParser(description='Advnced machine learning system', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-m', '--mnist', nargs='?', default='pathmnist', help='pathmnist or pneumoniamnist')
        parser.add_argument('-n', '--nettype', nargs='?', type=int, default=1, help='Self-defined neural network type')
        parser.add_argument('-l', '--lib', action='store_true', help='Use medmnist library')
        parser.add_argument('-dbg', '--debug', action='store_true', help='Print debug messages')
        return parser
    
    def run_pathmnist(self):
        mnist = Pathmnist(self.debug)
        mnist.data_loading(self.lib)
        mnist.define_model(self.nettype, mnist.n_channels, mnist.n_classes)
        mnist.train_model(self.nettype)
        mnist.test_model(self.nettype,'test')

    def run_pneumoniamnist(self):
        mnist = pnm.Pneumoniamnist()
        net = pnm.Net(in_channels=mnist.n_channels, num_classes=mnist.n_classes, type=self.nettype)
        net.hello()

def main():
    amls = Amls(Amls.argparser().parse_args())
    if (amls.mnist=='pathmnist'):
        amls.run_pathmnist()
    elif (amls.mnist=='pneumoniamnist'):
        amls.run_pneumoniamnist()
    else:
        Amls.argparser().print_help()

if __name__ == '__main__':
    main()
