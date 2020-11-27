'''
Author       : ZHP
Date         : 2020-11-23 16:27:26
LastEditors  : ZHP
LastEditTime : 2020-11-27 15:49:51
FilePath     : /Earlier_Project/get_mask.py
Description  : 通过LeNet提取feature map和标签，再经过co-localization得到rough mask,最后通过refinement得到mask
Copyright 2020 ZHP
'''
from rewrite.co_loco_visual import colocalization
import rewrite.refinement as refine
import test_LeNet as extractor
import os
import numpy as np
import torch
import cv2
from PIL import Image

FEATURE_MAP_SIZE = 25

def get_args():
    # Test settings
    parser = argparse.ArgumentParser(description='LeNet CME!')
    parser.add_argument('--testDir', default='/disk/dataset/test_poor/test', help='test image directory')
    parser.add_argument('--model_info', default='/disk/dataset/Earlier_Project/model_result/LeNet5/batch_4_lr_0.001/result.json', help='net info file')
    parser.add_argument('--save_dir_mask ', default='/disk/dataset/test_poor/Pre_wpy/Mask_1123', help='the dir to save mask image')
    parser.add_argument('--save_dir_cat', default='/disk/dataset/test_poor/Pre_wpy/OriginMask_1123', help='the dir to save stitched pictures')
    args = parser.parse_args()
    print('主要参数配置如下：\n')
    for key, value in args.__dict__.items():
        print(f'{key:^20} : {str(value):<}')
    return args


def generate_mask(img_path, model_info):
    '''
    description :  整合三个过程，生成mask
    param {
            img_path : 原图片路径
            model_info : 已训练好的网络信息文件路径，以'.json'结尾
    }
    return {
        mask : 生成的mask
        cat_mask_origin : 原图和mask拼接后的图片
    }
    '''
    origin_img = cv2.imread(img_path, 0)
    feature_map, label = extractor.test_one(model_info, img_path)                           # 通过LeNet提取特征图
    rough_mask = colocalization(feature_map, FEATURE_MAP_SIZE)                              # 通过协同定位得到rough mask
    mask, cat_mask_origin = refine.get_refine_mask(origin_img, np.array(rough_mask))        # 通过refine中图割优化rough mask
    return mask, cat_mask_origin


def Multi_level_folder(Parent_directory, model_info, save_dir_mask, save_dir_cat, start=None):
    '''
    description: 
    param {
        Parent_directory:最顶层目录，结尾不能是'/'
    }
    return {*}
    '''
    if start is None:
        start_key = Parent_directory
    else:
        start_key = start
    for path in os.listdir(Parent_directory):
        file_path = os.path.join(Parent_directory, path)
        if os.path.isdir(file_path):
            Multi_level_folder(file_path, model_info, save_dir_mask, save_dir_cat, start_key)
        elif os.path.isfile(file_path):
            if '.png' not in file_path:
                continue
            tailfix = file_path.split(start_key)[-1]
            figure_name = tailfix.split('/')[-1]
            mask, cat_img = generate_mask(file_path, model_info)
            save_mask = save_dir_mask + '/'.join(tailfix.split('/')[:-1])
            save_cat = save_dir_cat + '/'.join(tailfix.split('/')[:-1])
            if not (os.path.exists(save_mask) or os.path.exists(save_cat)):
                os.makedirs(save_mask)
                os.makedirs(save_cat)
            
            mask_file = os.path.join(save_mask, figure_name)
            cat_file = os.path.join(save_cat, figure_name)
            try:
                Image.fromarray(mask).save(mask_file)
                Image.fromarray(cat_img).save(cat_file)
            except:
                print(f'{file_path} 结果保存失败')
            else:
                print(f'{figure_name} 生成成功 \n mask保存为 : {mask_file} \n 拼接图片保存为 ： {cat_file}')


if __name__ == '__main__':
    args = get_args()
    # 多层目录文件测试
    Multi_level_folder(args.testDir, args.model_info, args.save_dir_mask, args.save_dir_cat)
    print('done..')