'''
Author       : ZHP
Date         : 2020-11-18 16:17:04
LastEditors  : ZHP
LastEditTime : 2020-11-19 16:11:18
FilePath     : /Earlier_Project/dataset.py
Description  : LeNet dataset (112x112)
Copyright 2020 ZHP
'''
# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torch
import json


class CMESet(data.Dataset):
    '''
    description: 用于LeNet网络的CME dataset
    param {
        root        : image folder
        lists       : 包含image名称和标签的字典文件，格式如：
                        "2011-01-01T00_11_41.png": 0
        transforms  : 指定的transforms
        test        : 是否为测试集
    }
    return {
        当test为True,仅返回图片(tensor)
        当test为False,返回图片(tensor)和标签(0/1)
    }
    '''

    def __init__(self, root, lists=None, transforms=None, test=False):
        super(CMESet, self).__init__()
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs.sort()
        self.imgs_list = imgs
        if not test:
            with open(lists, 'r') as f:
                lines = json.load(f)
            self.info = lines
            f.close()

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((112, 112)),  
                T.ToTensor()  
            ])
        else:
            self.transforms = transforms
        print('File number : {}\n'.format(len(self.imgs_list)))

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs_list[index]
        image = Image.open(img_path).convert('L')
        image = self.transforms(image)
        
        if self.test:
            return image
        else:
            name_idx = img_path.split('/')[-1]
            label = self.info[name_idx] 
            return image, label

    def __len__(self):
        return len(self.imgs_list)

if __name__ == '__main__':

    img_dir = '/disk/dataset/cme/pytorch/vgg/201101_modify/'
    label_info = '/disk/dataset/cme/pytorch/vgg/201101_modify_01_label.txt'
    dataset = CMESet(img_dir, label_info)
    train_loader = data.DataLoader(dataset, batch_size=4)
    for ii, (img, label) in enumerate(dataset):
        print(img.size(), img.float().mean(), label, torch.Tensor(label).dtype)
        if ii > 5:
            break
    for ii, (img, label) in enumerate(train_loader):
        print(img.size(), img.float().mean(), label, label.shape, label.dtype)
        if ii > 5:
            break
    
    test_dir = '/disk/dataset/cme_matting/test/test_ori/'
    test_set = CMESet(test_dir, test=True)
    for ii, img in enumerate(test_set):
        print(img.size(), img.float().mean())
        if ii > 5:
            break
