import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.autograd import Function, Variable
from torch.optim import Adam
from torchvision import datasets, transforms

class Unit(nn.Module):
    '''
    A basic unit in the SimpleNet. Performs a convolution, batch normalize, and a ReLu
    '''
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleNet,self).__init__()
        
        # The naming convention is weird because we used to have 4 times as many layers. We would have renamed these but we already saved the model 
        # so it was too late to rename :/
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.net = nn.Sequential(self.unit1, self.pool1, self.unit4, self.pool2, self.unit8, self.pool3, self.avgpool)

        self.fc = nn.Linear(in_features=128*2*2,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(output.shape[0],-1)
        output = self.fc(output)
        return output

def load_input(path, shuffle = True):
    '''
    load the images with root folder at path. Uses the ImageFolder to generate class labels for the images. Returns a trainloader object
    '''
    transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform = transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = shuffle)
    return train_loader

def train(model, train_loader, epochs):
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, prediction = torch.max(outputs.data, 1)
            running_loss += torch.sum(prediction != labels.data)
        else:
            print(f"Training loss: {running_loss/float(len(train_loader))}")

def predict_piece(model, patches):
    '''
    Take the model and the patches we obtained from the chessboard and output the classes of the patches
    '''
    # resize and transpose channels so that they can go into the neural net
    eval_patches = [cv2.resize(patch, (64, 64), interpolation = cv2.INTER_AREA) for patch in patches]
    eval_patches = [np.transpose(patch, (2, 0, 1)) for patch in eval_patches]

    # this is done to scale the intensity between 0 to 1 which is how pytorch loads its images
    patch_tensors = [torch.from_numpy(patch/255) for patch in eval_patches]

    # stack the images into a batch so that the model can predict them all at once
    patches_tensor = torch.stack(patch_tensors, dim = 0).float()
    model.eval()
    outputs = model(patches_tensor)
    _, predictions = torch.max(outputs, 1)
    return predictions.data