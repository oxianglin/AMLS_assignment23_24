import os
import sys
import argparse

sys.path.insert(0, f'{os.getcwd()}/A')
import pneumoniamnist as pnm

sys.path.insert(0, f'{os.getcwd()}/B')
import pathmnist as path

class Amls():
    def __init__(self, args:argparse.Namespace=()):
        super().__init__()
        self.mnist = args.mnist
        self.nettype = args.nettype

    @classmethod
    def argparser(cls): #angparser, my own AMLS member function/method 
        #argparse is site package with class ArgumentParser
        #ArgumentParser class has a member function called parse_args (returns value that we write into args)
        parser = argparse.ArgumentParser(description='Advnced machine learning system', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-m', '--mnist', nargs='?', default='pathmnist', help='pathmnist or pneumoniamnist')
        parser.add_argument('-n', '--nettype', nargs='?', type=int, default=1, help='Self-defined neural network type')
        return parser
    
    def run_pathmnist(self):
        mnist = path.Pathmnist()
        net = path.Net(in_channels=mnist.n_channels, num_classes=mnist.n_classes, type=self.nettype)
        net.hello()

    def run_pneumoniamnist(self):
        mnist = pnm.Pneumoniamnist()
        net = pnm.Net(in_channels=mnist.n_channels, num_classes=mnist.n_classes, type=self.nettype)
        net.hello()

def print_help():
    Amls.argparser().print_help()

def main():
    amls = Amls(Amls.argparser().parse_args())
    if (amls.mnist=='pathmnist'):
        amls.run_pathmnist()
    elif (amls.mnist=='pneumoniamnist'):
        amls.run_pneumoniamnist()
    else:
        print_help()
        

if __name__ == '__main__':
    main()
