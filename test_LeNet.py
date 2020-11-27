'''
Author       : ZHP
Date         : 2020-11-19 10:57:14
LastEditors  : ZHP
LastEditTime : 2020-11-27 16:44:04
FilePath     : /Earlier_Project/test_LeNet.py
Description  : 利用训练好的LeNet模型生成feature map和分类结果
Copyright 2020 ZHP
'''
# coding: utf-8
import os
import torch
import models
import json
import copy
import numpy as np
from torchvision import transforms
import time
from dataset import CMESet
from PIL import Image
import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    # Test settings
    parser = argparse.ArgumentParser(description='LeNet CME!')
    parser.add_argument('--testDir', default='./temporary_folder/test_ori/', help='test image directory')
    parser.add_argument('--info_path', default='./model_result/LeNet5/batch_4_lr_0.001/result.json', help='net info file')
    parser.add_argument('--saveDir', default='./temporary_folder/feature_maps/', help='feature maps save dir')
    parser.add_argument('--result_path', default='./update_label.json', help='test image label')
    args = parser.parse_args()
    print('主要参数配置如下：\n')
    for key, value in args.__dict__.items():
        print(f'{key:^20} : {str(value):<}')
    return args


@torch.no_grad()
def test(info_path, test_dir, save_dir, result_path):
    '''
    description: 通过已训练的模型生成测试结果
                包括中间层提取的feature map，以及网络预测的标签
    param {
        info_path : 网络信息文件path,以'.json'结尾
        test_img_path : 测试图片文件夹
        save_dir : 测试结果保存位置
        result_path : 测试结果写入的json文件
    }
    return {*}
    '''
    with open(info_path, 'r') as f:
        info = json.load(f)
        f.close()
    
    model = getattr(models, info['model_name'])(grayscale=True, test=True)
    model.to(device).eval()
    model_state = torch.load(info['best_model_path'])
    model.load_state_dict(model_state, strict=True)


    transform = transforms.Compose([
        transforms.Resize(info['size'], 0),
        transforms.ToTensor()
    ])
    test_set = CMESet(test_dir, transforms=transform, test=True)
    out_label = {}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, image in enumerate(test_set):
        img_name = test_set.imgs_list[i]
        image = image.to(device)
        feature_map, y_pre = model(image.unsqueeze(0))
        feature_map = feature_map.squeeze().reshape(50, 25*25).transpose(1, 0)
        feature_map = feature_map.cpu().detach().clone().numpy()
        feature_map_file = save_dir + img_name.split('/')[-1].split('.')[0] + '.npy'
        np.save(feature_map_file, feature_map)

        label = y_pre[0].argmax().item()
        out_label[img_name] = label
    
    # 预测结果写入
    # 这里想永远只留一个Json保存所有图片预测标签，故在旧的文件中添加新的标签
    with open(result_path, 'r') as f:
        old_file = json.load(f)
    f.close()
    old_file.update(out_label)
    with open(result_path, 'w') as f:
        json.dump(old_file, f, indent=4)
    f.close()
    print('extractor feature map hold in {0}\npredict label hold in {1}'.format(save_dir, result_path))


# 测试单张图片
@torch.no_grad()
def test_one(info_path, test_img_path):
    '''
    description: 通过已训练的模型生成测试结果
                包括中间层提取的feature map，以及网络预测的标签
    param {
        info_path : 网络信息文件path,以'.json'结尾
        test_img_path : 测试图片path,以'.png'结尾
    }
    return {*}
    '''
    with open(info_path, 'r') as f:
        info = json.load(f)
        f.close()
    
    model = getattr(models, info['model_name'])(grayscale=True, test=True)
    model.to(device).eval()
    try:
        model_state = torch.load(info['best_model_path'])
        model.load_state_dict(model_state, strict=True)
    except:
        pass
    else:
        print('{} 导入成功！'.format(info['best_model_path']))
    transform = transforms.Compose([
        transforms.Resize(info['size'], 0),
        transforms.ToTensor()
    ])
    img_name = test_img_path.split('/')[-1].split('.')[0]
    image = Image.open(test_img_path).convert('L')
    image = transform(image)
    image = image.to(device)
    
    feature_map, y_pre = model(image.unsqueeze(0))
    feature_map = feature_map.squeeze().reshape(50, 25*25).transpose(1, 0)
    feature_map = feature_map.cpu().detach().clone().numpy()

    label = y_pre[0].argmax().item()
    return feature_map, label


if __name__ == "__main__":
    global args
    args = get_args() 
    test(args.info_path, args.testDir, args.saveDir, args.result_path)
