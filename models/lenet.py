import torch
import torch.nn as nn
import numpy as np
# import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

# import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
# from datasetLenet import DateSet

class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 6*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6*in_channels, 16*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5*in_channels, 120*in_channels),
            nn.Tanh(),
            nn.Linear(120*in_channels, 84*in_channels),
            nn.Tanh(),
            nn.Linear(84*in_channels, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        logits = self.classifier(x)
        print(logits.shape)
        probas = F.softmax(logits, dim=1)
        print(probas.shape)
        return logits, probas


if __name__ == '__main__':
    model = LeNet5(2)
    # summary(model.cuda(), (3,512,512))
    data = torch.randn(1,3,224,224)
    model(data)