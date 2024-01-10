from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import medmnist
from medmnist import INFO
from sklearn.metrics import accuracy_score

DEBUG = False

def dbg_print(s):
    if DEBUG:
        print(s, flush=True)

class Net(nn.Module):
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
            dbg_print(f'Before layer1, {x.size()}')
            x = self.layer1(x)
            dbg_print(f'After layer1, {x.size()}')
            x = self.layer2(x)
            dbg_print(f'After layer2, {x.size()}')
            x = self.layer3(x)
            dbg_print(f'After layer3, {x.size()}')
            x = self.layer4(x)
            dbg_print(f'After layer4, {x.size()}')
            x = self.layer5(x)
            dbg_print(f'After layer5, {x.size()}')
            x = x.view(x.size(0), -1)
            dbg_print(f'After view, {x.size()}')
            x = self.fc(x)
            dbg_print(f'After fc, {x.size()}')
            return x

class data_set(data.Dataset):
    data_file = ''
    def __init__(self, split, transform=None, target_transform=None, as_rgb=False): #Initialise dataset.
        if (self.data_file == ''):
            self.search_data()
        npz_file = np.load(self.data_file)
        if (DEBUG and split == 'train'):
            lst = npz_file.files
            dbg_print('pathmnist.npz contans:')
            for item in lst:
                dbg_print(item)
            for item in lst:    
                if (item == 'train_images'):
                    dbg_print(f'npz_file[{item}].shape = {npz_file[item].shape}')
                    dbg_print(f'npz_file[{item}][0].shape = {npz_file[item][0].shape}')
                    dbg_print(f'npz_file[{item}][0][0].shape = {npz_file[item][0][0].shape}')
                    dbg_print(f'npz_file[{item}][0][0] = {npz_file[item][0][0]}')
                    dbg_print(f'npz_file[{item}][0][0][0].shape = {npz_file[item][0][0][0].shape}')
                    dbg_print(f'npz_file[{item}][0][0][0] = {npz_file[item][0][0][0]}')
        if (DEBUG and split == 'val'):
            lst = npz_file.files
            for item in lst:    
                if (item == 'val_images'):
                    dbg_print(f'npz_file[{item}].shape = {npz_file[item].shape}')
        if (DEBUG and split == 'test'):
            lst = npz_file.files
            for item in lst:    
                if (item == 'test_images'):
                    dbg_print(f'npz_file[{item}].shape = {npz_file[item].shape}')
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
  
    def __len__(self): #For DataLoader class to get the number of items in the dataset.
        return self.imgs.shape[0]
  
    def __getitem__(self, index): #For DataLoader class to get the items.
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
        dir = os.path.dirname(__file__) #<The directory of this file> which is <upper directories/amls/B>.
        ds = dir.split(os.sep)
        ds.pop(len(ds)-1) #Move one level up in directory.
        dir = os.sep.join(ds)
        data_set.data_file = f'{dir}{os.sep}Datasets{os.sep}PathMNIST{os.sep}pathmnist.npz'

class Pathmnist():
    def __init__(self, debug):
        global DEBUG
        DEBUG = debug
        info = INFO['pathmnist']
        self.task = info['task']
        self.n_channels = info['n_channels']
        self.n_classes = len(info['label'])
        self.data_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])
        print('Use code extracted minimally from medmnist library for datatset processing')
        print('Dataset objects to be initiated from data_set class contain pathmnist data')
        self.train_dataset = data_set(split='train', transform=self.data_transform)
        self.val_dataset = data_set(split='val', transform=self.data_transform)
        self.test_dataset = data_set(split='test', transform=self.data_transform)

    def data_loading(self, batch_size=128):
        print('The data to be encapsulated into DataLoader form')
        self.train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = data.DataLoader(dataset=self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = data.DataLoader(dataset=self.test_dataset, batch_size=2*batch_size, shuffle=False)
    
    def define_model(self, nettype, n_channels, n_classes, learningrate=0.001):
        self.model = Net(in_channels=n_channels, num_classes=n_classes, nettype=nettype)   
        print('Define loss function and optimiser')
        if self.task == "multi-label, binary-class":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learningrate, momentum=0.9)

    def train_model(self, nettype, epoch=3):
        self.val_acc = []
        self.saved_model = []
        self.data_loading()
        for epoch in range(epoch):
            print(f'Train the model with train set at epoch {epoch}')
            self.model.train()
            for inputs, targets in tqdm(self.train_loader):
                # forward + backward + optimise
                self.optimizer.zero_grad()
                outputs = self.model(inputs, nettype)                
                if self.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = self.criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long() #Remove axes of length one then convert to int64.
                    loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            self.val_acc.append(self.test_model(nettype, 'val'))
            self.saved_model.append(f'{os.getcwd()}{os.sep}saved_model_{epoch}')
            torch.save(self.model, f'{os.getcwd()}{os.sep}saved_model_{epoch}')
            print(f'Model saved in {os.getcwd()}{os.sep}saved_model_{epoch}')

    def test_model(self, nettype, split):
        if (split == 'test'):
            idx = self.val_acc.index(max(self.val_acc))
            self.model = torch.load(f'{os.getcwd()}{os.sep}saved_model_{idx}')
            print(f'Model loaded from {os.getcwd()}{os.sep}saved_model_{idx}')
            print('Test the model with test set')
        elif (split == 'val'):
            print('Evaluate the model with val set')
        self.model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        data_loader = self.val_loader if split == 'val' else self.test_loader
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader):
                outputs = self.model(inputs, nettype)
                if self.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
            y_true = y_true.numpy().squeeze() #Convert tensor to numpy then remove axes of length one.
            y_score = y_score.numpy().squeeze()
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
            print('%s acc:%.3f' % (split, accuracy))
            return accuracy
