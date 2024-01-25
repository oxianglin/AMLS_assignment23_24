from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
from medmnist import INFO
from sklearn.metrics import accuracy_score

class Net(nn.Module): #Adopted from memdmnist sample code.
    def __init__(self, in_channels, num_classes, nettype):
        super(Net, self).__init__()
        if (nettype==1):
            print('Define CNN model')
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

    def forward(self, x, nettype):
        if (nettype==1):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

class data_set(data.Dataset): #Adopted from medmnist site-package.
    data_file = ''
    def __init__(self, split, transform=None, target_transform=None, as_rgb=False): #Initialize dataset
        if (self.data_file == ''):
            self.search_data()
        npz_file = np.load(self.data_file)

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb
        if self.split == 'train':
            self.imgs = npz_file['train_images']
            self.labels = npz_file['train_labels']
        elif self.split == 'val':
            self.imgs = npz_file['val_images']
            self.labels = npz_file['val_labels']
        elif self.split == 'test':
            self.imgs = npz_file['test_images']
            self.labels = npz_file['test_labels']
        else:
            raise ValueError
  
    def __len__(self): #For DataLoader class to get the number of item in dataset.
        return self.imgs.shape[0]
  
    def __getitem__(self, index): #For DataLoader class to get the item.
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img) #Create image object from numpy array.
        if self.as_rgb:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def search_data(self):
        dir = os.path.dirname(__file__) #<The directory name of this file>.
        ds = dir.split(os.sep)
        ds.pop(len(ds)-1) #Move one level up to AMLS directory.
        dir = os.sep.join(ds)
        data_set.data_file = f'{dir}{os.sep}Datasets{os.sep}PneumoniaMNIST{os.sep}pneumoniamnist.npz'

class Pneumoniamnist(): #Adopted from medmnist sample code.
    def __init__(self, config):
        info = INFO['pneumoniamnist']
        self.task = info['task']
        self.n_channels = info['n_channels']
        self.n_classes = len(info['label'])
        self.seed = config['seed']
        self.id = config['id']
        self.nettype = config['nettype']
        self.batch_size = config['batch_size']
        self.epoch = config['epoch']
        self.lr = config['lr']
        self.momentum = config['momentum']
        self.data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])
        torch.manual_seed(self.seed)
        print('Dataset objects to be instantiated from data_set class containing pneumoniamnist data')
        self.train_dataset = data_set(split='train', transform=self.data_transform)
        self.val_dataset = data_set(split='val', transform=self.data_transform)
        self.test_dataset = data_set(split='test', transform=self.data_transform)
    
    def define_model(self):
        self.model = Net(in_channels=self.n_channels, num_classes=self.n_classes, nettype=self.nettype)   
        print('Loss function and optimiser are defined.')
        if self.task == "multi-label, binary-class":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    def train_model(self):
        val_acc = []
        val_loss = []
        fit_count = 0
        self.saved_model = []
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = data.DataLoader(dataset=self.test_dataset, batch_size=2*self.batch_size, shuffle=False)
        for epoch in range(self.epoch):
            print(f'Train the model using train set at epoch {epoch}.')
            self.model.train()
            for inputs, targets in tqdm(self.train_loader):
                # Forward + backward + optimise.
                self.optimizer.zero_grad()
                outputs = self.model(inputs, self.nettype)                
                if self.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = self.criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long() #Remove axes of length one then convert to int64
                    loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            acc, loss = self.test_model('val')
            val_acc.append(acc)
            val_loss.append(loss)
            print('val acc:%.3f, val loss:%.3f' % (val_acc[epoch], val_loss[epoch]))
            if epoch>0 and (val_acc[epoch]-val_acc[epoch-1]<0.001) and (val_acc[epoch]<0.85 or val_acc[epoch]>0.970):
                print('Stopping criteria met.')
                break
            if val_acc[epoch] > 0.96:
                fit_count += 1
            if fit_count > (self.epoch/2):
                print(f'Stopping criteria met. Fit count = {fit_count}')
                break
        self.saved_model.append(f'{os.getcwd()}{os.sep}saved_model_{self.id}.')
        torch.save(self.model, f'{os.getcwd()}{os.sep}saved_model_{self.id}.')
        print(f'Model saved in {os.getcwd()}{os.sep}saved_model_{self.id}.')
        return val_acc, val_loss

    def test_model(self, split, id=1):
        if (split == 'test'):
            self.model = torch.load(f'{os.getcwd()}{os.sep}saved_model_{id}.')
            print(f'Model loaded from {os.getcwd()}{os.sep}saved_model_{id}.')
            print('Test the model with test set.')
        elif (split == 'val'):
            print('Evaluate the model with val set.')
        self.model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        data_loader = self.val_loader if split == 'val' else self.test_loader
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader):
                outputs = self.model(inputs, self.nettype)
                if self.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = self.criterion(outputs, targets)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long() #Remove axes of length one, then convert to int64.
                    loss = self.criterion(outputs, targets)
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                total_loss += loss.cpu().numpy() #Loss calculation adopted from https://www.kaggle.com/code/abhishekrathi09/pytorch-basics-part-6-hyperparameter-tuning
                steps += 1
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
            y_true = y_true.numpy().squeeze() #Convert tensor to numpy, then remove axes of length one.
            y_score = y_score.numpy()
            if self.task == 'multi-label, binary-class':
                y_pre = y_score > 0.5
                acc = 0
                for label in range(y_true.shape[1]):
                    label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
                    acc += label_acc
                accuracy = acc / y_true.shape[1]
            elif self.task == 'binary-class':
                if y_score.ndim == 2:
                    y_score = y_score[:, -1]
                else:
                    assert y_score.ndim == 1
                accuracy = accuracy_score(y_true, y_score > 0.5)
            else:
                accuracy = accuracy_score(y_true, np.argmax(y_score, axis=-1)) #np.argmax returns the indices of the maximum values along the last axis.
            return accuracy, total_loss/steps
