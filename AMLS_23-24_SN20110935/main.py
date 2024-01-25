import os
import sys
import argparse
import random
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'A')) #Ensure <the directory name of this file>/A is the first path in sys.path list for import to find module pneumoniamnist.py.
from pneumoniamnist import Pneumoniamnist
from knn_pnm import knn_pnm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'B')) #Ensure <the directory name of this file>/B is the first path in sys.path list for import to find module pathmnist.py.
from pathmnist import Pathmnist
from knn_path import knn_path

class Amls():

    def __init__(self, args:argparse.Namespace=()):
        super().__init__()

        self.mnist = args.mnist
        self.id = args.id
        self.knn = args.knn

        #Some contents are not hyperparameters (e.g. nettype).
        #id for identification, to be used when saving model after each training, and loading a model before testing.
        if (self.mnist=='pneumoniamnist'):
            self.hyperparameter = [{'id':1 if not self.id else self.id, 'nettype':1, 'seed':106, 'batch_size':64, 'epoch':8, 'lr':0.05, 'momentum':0.8}, 
                                    {'id':2 if not self.id else self.id+1, 'nettype':1, 'seed':random.randrange(1,sys.maxsize), 'batch_size':random.choice([32,64,96,128]), 'epoch':random.choice([5,6,7,8,9,10]), 'lr':random.choice([0.001,0.002,0.003,0.005,0.008,0.01,0.05,0.1]), 'momentum':random.choice([0.5,0.6,0.7,0.8,0.9])},
                                    {'id':3 if not self.id else self.id+2, 'nettype':1, 'seed':random.randrange(1,sys.maxsize), 'batch_size':random.choice([32,64,96,128]), 'epoch':random.choice([5,6,7,8,9,10]), 'lr':random.choice([0.001,0.002,0.003,0.005,0.008,0.01,0.05,0.1]), 'momentum':random.choice([0.5,0.6,0.7,0.8,0.9])}]                        
#                                   {'id':1 if not self.id else self.id, 'nettype':1, 'seed':106, 'batch_size':64, 'epoch':8, 'lr':0.05, 'momentum':0.92},
#                                   {'id':1 if not self.id else self.id, 'nettype':1, 'seed':106, 'batch_size':64, 'epoch':8, 'lr':0.05, 'momentum':0.9},
#                                   {'id':1 if not self.id else self.id, 'nettype':1, 'seed':106, 'batch_size':64, 'epoch':8, 'lr':0.05, 'momentum':0.9},
#                                   {'id':1 if not self.id else self.id, 'nettype':1, 'seed':106, 'batch_size':64, 'epoch':8, 'lr':0.05, 'momentum':1.0}]

#            self.hyperparameter = [
#                        {'id':1 if not self.id else self.id, 'nettype':1, 'seed':106, 'batch_size':64, 'epoch':15, 'lr':0.05, 'momentum':0.9}
#                        ]
        else:
            self.hyperparameter = [{'id':11 if not self.id else self.id, 'nettype':1, 'seed':34, 'batch_size':128, 'epoch':6, 'lr':0.003, 'momentum':0.9}, 
                        {'id':12 if not self.id else self.id+1, 'nettype':1, 'seed':random.randrange(1,sys.maxsize), 'batch_size':random.choice([64,96,128,192,256]), 'epoch':random.choice([5,6,7,8,9]), 'lr':random.choice([0.001,0.002,0.003,0.005,0.008,0.01,0.05,0.1]), 'momentum':random.choice([0.5,0.6,0.7,0.8,0.9])},
                        {'id':13 if not self.id else self.id+2, 'nettype':1, 'seed':random.randrange(1,sys.maxsize), 'batch_size':random.choice([64,96,128,192,256]), 'epoch':random.choice([5,6,7,8,9]), 'lr':random.choice([0.001,0.002,0.003,0.005,0.008,0.01,0.05,0.1]), 'momentum':random.choice([0.5,0.6,0.7,0.8,0.9])}]
#            self.hyperparameter = [
#                        {'id':11 if not self.id else self.id, 'nettype':1, 'seed':34, 'batch_size':128, 'epoch':6, 'lr':0.003, 'momentum':0.9}
#                        ]

        self.val_accuracy = []
        self.val_loss = []
        self.test_accuracy = []
        self.test_loss = []
        self.maxlen = 0
        self.legend = []
        self.best_trial = []

    @classmethod
    def argparser(cls):
        parser = argparse.ArgumentParser(description='Applied Machine Learning Systems', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-m', '--mnist', nargs='?', default='pneumoniamnist', help='pathmnist or pneumoniamnist')
        parser.add_argument('-id', '--id', nargs='?', type=int, default=0, help='Set to 1 or above to be appended to the name of model to be saved')
        parser.add_argument('-knn', '--knn', action='store_true', help='Use knn model')
        return parser
    
    def plot(self, data, legend): #https://maschituts.com/how-to-plot-a-list-in-python/
        self.legend.append(legend)
        if len(data)>self.maxlen:
            self.maxlen = len(data)
        plt.plot(data, marker='o', mfc='pink' ) #Plot the data.

    def show_plot(self):
        plt.xticks(range(0, self.maxlen+1, 1)) #Set the tick frequency on x-axis.
        plt.ylabel('val_accuracy') #Set the label for y axis.
        plt.xlabel('epoch') #Set the label for x-axis.
        plt.title("Learning curve") #Set the title of the graph.
        plt.legend(self.legend)
        plt.show() #Display the graph.

    def run_pathmnist(self):
        for trial in range(len(self.hyperparameter)): 
            print(f'TRIAL {trial}: train, evaluation and test.')
            print(f"Chosen hyperparameter: seed={self.hyperparameter[trial]['seed']}, batch_size={self.hyperparameter[trial]['batch_size']}, epoch={self.hyperparameter[trial]['epoch']}, lr={self.hyperparameter[trial]['lr']}, momentum={self.hyperparameter[trial]['momentum']}")
            mnist = Pathmnist(config=self.hyperparameter[trial])
            mnist.define_model()
            acc, loss = mnist.train_model()
            self.val_accuracy.append(acc)
            self.val_loss.append(loss)
            print(f'val accuracy ={self.val_accuracy[trial]}, val loss = {self.val_loss[trial]}')
            self.plot(self.val_accuracy[trial], f'trial {trial}')
            test_accuracy, test_loss = mnist.test_model('test', self.hyperparameter[trial]['id'])
            self.test_accuracy.append(test_accuracy)
            self.test_loss.append(test_loss)
            print('test accuracy = %.3f, test loss = %.3f' % (self.test_accuracy[trial], self.test_loss[trial]))
        print('Summary of all trials:')
        idx = self.test_accuracy.index(max(self.test_accuracy))
        print(f'Best trial is trial {idx}.')
        print(f"Best hyperparameter: seed={self.hyperparameter[idx]['seed']}, batch_size={self.hyperparameter[idx]['batch_size']}, epoch={self.hyperparameter[idx]['epoch']}, lr={self.hyperparameter[idx]['lr']}, momentum={self.hyperparameter[idx]['momentum']}")
        print('test accuracy = %.3f, test loss = %.3f' % (self.test_accuracy[idx], self.test_loss[idx]))
        self.show_plot()

    def run_pneumoniamnist(self):
        for trial in range(len(self.hyperparameter)): 
            print(f'TRIAL {trial}: train, evaluation and test.')
            print(f"Chosen hyperparameter: seed={self.hyperparameter[trial]['seed']}, batch_size={self.hyperparameter[trial]['batch_size']}, epoch={self.hyperparameter[trial]['epoch']}, lr={self.hyperparameter[trial]['lr']}, momentum={self.hyperparameter[trial]['momentum']}")
            mnist = Pneumoniamnist(config=self.hyperparameter[trial])
            mnist.define_model()
            acc, loss = mnist.train_model()
            self.val_accuracy.append(acc)
            self.val_loss.append(loss)
            print(f'val accuracy ={self.val_accuracy[trial]}, val loss = {self.val_loss[trial]}')
            self.plot(self.val_accuracy[trial], f'trial {trial}')
            #self.plot(self.val_accuracy[trial], f'momentum = {self.hyperparameter[trial]["momentum"]}')
            test_accuracy, test_loss = mnist.test_model('test', self.hyperparameter[trial]['id'])
            self.test_accuracy.append(test_accuracy)
            self.test_loss.append(test_loss)
            print('test accuracy = %.3f, test loss = %.3f' % (self.test_accuracy[trial], self.test_loss[trial]))
        print('Summary of all trials:')
        idx = self.test_accuracy.index(max(self.test_accuracy))
        print(f'Best trial is trial {idx}.')
        print(f"Best hyperparameter: seed={self.hyperparameter[idx]['seed']}, batch_size={self.hyperparameter[idx]['batch_size']}, epoch={self.hyperparameter[idx]['epoch']}, lr={self.hyperparameter[idx]['lr']}, momentum={self.hyperparameter[idx]['momentum']}")
        print('test accuracy = %.3f, test loss = %.3f' % (self.test_accuracy[idx], self.test_loss[idx]))
        self.show_plot()
        
def main():
    amls = Amls(Amls.argparser().parse_args())
    if (amls.mnist=='pathmnist'): #Apply -m pathmnist as argument to get here.
        if (amls.knn):
            knn_path()
        else:
            amls.run_pathmnist()
    elif (amls.mnist=='pneumoniamnist'): #Default.
        if (amls.knn):
            knn_pnm()
        else:
            amls.run_pneumoniamnist()
    else:
        Amls.argparser().print_help()

if __name__ == '__main__':
    main()
