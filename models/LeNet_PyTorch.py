'''
Author       : ZHP
Date         : 2020-11-18 19:49:30
LastEditors  : ZHP
LastEditTime : 2020-11-19 11:25:43
FilePath     : /Earlier_Project/models/LeNet_PyTorch.py
Description  : 王鹏宇学长论文中LeNet的PyTorch实现
               论文：
               Wang, Pengyu,Zhang, Yan,Feng, Li,et al. A New Automatic Tool for CME Detection and Tracking 
               with Machine-learning Techniques[J]. ASTROPHYSICAL JOURNAL SUPPLEMENT SERIES,2019,244(1):11.
'''

'''
Description  : 王鹏宇学长论文中LeNet的PyTorch实现
               论文：
               Wang, Pengyu,Zhang, Yan,Feng, Li,et al. A New Automatic Tool for CME Detection and Tracking 
               with Machine-learning Techniques[J]. ASTROPHYSICAL JOURNAL SUPPLEMENT SERIES,2019,244(1):11.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet5(nn.Module):

    def __init__(self, num_classes=2, grayscale=False, test=False):
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes
        self.test = test
        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(50*25*25, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )


    def forward(self, x):
        x_data = self.features(x)
        x = x_data.view(x_data.size(0), -1)
        logits = self.classifier(x)
        probas = F.log_softmax(logits, dim=1)
        if self.test:
            return x_data, probas
        else:
            return probas
    



if __name__ == '__main__':
    model = LeNet5(2, grayscale=True)
    model_test = LeNet5(2, grayscale=True, test=True)
    # summary(model.cuda(), (1, 112, 112))
    data = torch.randn(4, 1, 112, 112)
    y_pre = model(data)
    print(y_pre, y_pre.shape)

    feature_map, y_pre = model_test(data)
    print(feature_map.shape, y_pre.shape)