import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

class Net(nn.Module):
    def __init__(self, in_channels, num_classes, type):
        super(Net, self).__init__()

        if (type==1):
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3),
                nn.BatchNorm2d(16),
                nn.ReLU())

            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.layer3 = nn.Sequential(
                nn.Conv2d(16, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU())
            
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU())

            self.layer5 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.fc = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes))

    def forward(self, x, type):
        if (type==1):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    def hello(self):
        print('hello pneumoniamnist')

class Pneumoniamnist():
    def __init__(self):
        info = INFO['pneumoniamnist']
        self.n_channels = info['n_channels']
        self.n_classes = len(info['label'])
        self.data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])
