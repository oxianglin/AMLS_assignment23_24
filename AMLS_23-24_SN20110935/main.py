import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'A')) #To ensure that <the name of the directory that contains this file>/A is the first path in sys.path list for import to find module pneumoniamnist.py
from pneumoniamnist import Pneumoniamnist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'B')) #To ensure that <the name of the directory that contains this file>/B is the first path in sys.path list for import to find module pathmnist.py
from pathmnist import Pathmnist

class Amls():
    def __init__(self, args:argparse.Namespace=()):
        super().__init__()
        self.mnist = args.mnist
        self.nettype = args.nettype
        if (self.nettype > 1): #To be changed based on the number of nettypes supported
            self.nettype = 1
        self.debug = args.debug

    @classmethod
    def argparser(cls): #argparser is my own AMLS member function/method 
        #argparse is site package with class ArgumentParser
        #ArgumentParser class has a member function called parse_args (returns value that we write into args)
        parser = argparse.ArgumentParser(description='Applied Machine Learning Systems', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-m', '--mnist', nargs='?', default='pathmnist', help='pathmnist or pneumoniamnist')
        parser.add_argument('-n', '--nettype', nargs='?', type=int, default=1, help='Self-defined neural network type')
        parser.add_argument('-dbg', '--debug', action='store_true', help='Print debug messages')
        return parser
    
    def run_pathmnist(self):
        mnist = Pathmnist(self.debug) #Instantiate an object of class Pathmnist defined in B/pathmnist.py.
        mnist.define_model(self.nettype, mnist.n_channels, mnist.n_classes)
        mnist.train_model(self.nettype)
        mnist.test_model(self.nettype, 'test')

    def run_pneumoniamnist(self):
        mnist = Pneumoniamnist(self.debug) #Instantiate an object of class Pneumoniamnist defined in A/pneumoniamnist.py.
        mnist.define_model(self.nettype, mnist.n_channels, mnist.n_classes)
        mnist.train_model(self.nettype)
        mnist.test_model(self.nettype, 'test')

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
