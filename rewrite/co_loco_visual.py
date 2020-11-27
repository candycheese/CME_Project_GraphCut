'''
Author       : ZHP
Date         : 2020-11-23 15:39:16
LastEditors  : ZHP
LastEditTime : 2020-11-27 16:39:39
FilePath     : /Earlier_Project/rewrite/co_loco_visual.py
Description  : CME Region co-localization
Copyright 2020 ZHP
'''
import os
import numpy as np
import torch
from sklearn.decomposition import PCA
import torch.nn.functional as F
from PIL import Image

IMAGE_SIZE = 512

def get_feat_map(dir, picname, feature_map_size):
    '''
    description: 获取由CNN提取的feature map
    param {
        dir:feature map 的.npy文件夹
        picname : feature map文件名，以'.npy'结尾
    }
    return {
        相应的feature map,形状为(25*25, 50)
    }
    '''
    feat_map=np.load(dir+picname)       #修改np.load为np.asarray  .txt为.npy  
    # print(feat_map.shape)
    feat=feat_map.reshape(feature_map_size * feature_map_size, 50)
    return feat


def write_one_pca_feature_map(OUT_IMAGE_DIR, name, feature_map):
    '''
    description: 
    param {*}
    return {*}
    '''
    Zconcat_target = Image.new('RGB', (IMAGE_SIZE*2, IMAGE_SIZE))
    zero=np.zeros((GLOBAL_FEATURE_MAP_SIZE, GLOBAL_FEATURE_MAP_SIZE),dtype = np.uint8)  # 25*25
    f = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) # 归一化
    f = np.int32(f>0.0)*f
    # 保存归一化后feature
    co_feature_path = OUT_IMAGE_DIR+name.split('.')[0]+'_cofeature.npy'
    np.save(co_feature_path, f)
    
    # 合成图片
    r = Image.fromarray(f.reshape(GLOBAL_FEATURE_MAP_SIZE,GLOBAL_FEATURE_MAP_SIZE)*255).convert('L')
    g = Image.fromarray(zero).convert('L')
    b = Image.fromarray(zero).convert('L')
    high_light=Image.merge("RGB", (r, g, b))
    high_light=high_light.resize((IMAGE_SIZE,IMAGE_SIZE))
    origin = Image.open(IMAGE_DIR + name).convert('RGB')
    Zconcat_target.paste(high_light, (IMAGE_SIZE,0))
    Zconcat_target.paste(origin, (0,0))
    Zconcat_target.save(OUT_IMAGE_DIR + name)
    print('{}保存成功'.format(OUT_IMAGE_DIR + name))


def get_feature_map_img(feature_map, feature_map_size):
    '''
    description: 
    param {
        feature_map : ndarray,shape为(IMAGE_SIZE*IMAGE_SIZE,)
        feature_map_size ： 
    }
    return {
        rough mask,PIL.Image对象，转成ndarray形状为(IMAGE_SIZE, IMAGE_SIZE, 3)
    }
    '''
    Zconcat_target = Image.new('RGB', (IMAGE_SIZE*2, IMAGE_SIZE))
    zero=np.zeros((feature_map_size, feature_map_size),dtype = np.uint8)  # 25*25
    f = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min()) # 归一化
    f = np.int32(f>0.0)*f


    r = Image.fromarray(f.reshape(feature_map_size, feature_map_size)*255).convert('L')
    g = Image.fromarray(zero).convert('L')
    b = Image.fromarray(zero).convert('L')
    high_light=Image.merge("RGB", (r, g, b))
    high_light=high_light.resize((IMAGE_SIZE,IMAGE_SIZE))
    
    '''
    rough_mask = np.expand_dims(f.reshape(feature_map_size,feature_map_size)*255, axis=2)
    g_b = np.zeros((feature_map_size, feature_map_size, 2))
    rough_mask = np.concatenate([rough_mask, g_b], axis=2)
    
    high_light=Image.fromarray(np.uint8(rough_mask))
    high_light=high_light.resize((IMAGE_SIZE,IMAGE_SIZE))
    '''
    return high_light


def define_trainer_img(feature_map, feature_map_size):
    '''
    description: 对所有feature map进行PCA降维
    param {
        feature_map : 通过CNN提取的特征，ndarray
        feature_map_size ： 理论上应为sqrt(feature_map.shape[0])
    }
    return {所有feature map的PCA和25*25个feature map的PCA}
    '''
    feature_map_PCA = np.expand_dims(feature_map, axis=0).transpose(1, 0, 2)
    pca_all = PCA(n_components=1)   # PCA:原始数据X, X=MN --> M=XN^T,n_components:PCA降维后的特征维度数目
    pca_all.fit(feature_map_PCA.reshape(-1, 50))  # pca_all.fit(X)-->.transform(x)返回降维后的数据M，合并则为fit_transform(x)
    pca_trainer_list = []
    for single_pixel in range(feature_map_size, feature_map_size):
        pca = PCA(n_components=1)
        pca.fit(feature_map_PCA[single_pixel])   # # pca.fit((1,50))
        pca_trainer_list.append(pca)
    # 返回所有feature map的PCA和25*25个feature map的PCA
    return pca_all, pca_trainer_list


def colocalization(feature_map, feature_map_size):
    '''
    description: 给定的feature map,生成粗糙的mask
    param {
        feature_map : 通过CNN提取的特征，ndarray
        feature_map_size ： 理论上应为sqrt(feature_map.shape[0])
    }
    return {
        rough mask,PIL.Image对象，转成ndarray形状为(IMAGE_SIZE, IMAGE_SIZE, 3)
    }
    '''
    all_pca, pca_list = define_trainer_img(feature_map, feature_map_size)
    feature_map_PCA = np.expand_dims(feature_map, axis=0).transpose(1, 0, 2)
    out_pca_list = []
    for single_pixel in range(feature_map_size * feature_map_size):
        # .transform(x) x.shape-->(N, 50) -->get M (M.shape(N,1)),M.dtype-->(ndarray)
        out_pca = all_pca.transform(feature_map_PCA[single_pixel])    # out_pca 是ndarray,shape(N,1)
        out_pca_list.append(out_pca)
    out_pca = np.asarray(out_pca_list).reshape(feature_map_size * feature_map_size, -1).transpose(1,0).squeeze()
    # print(out_pca.shape) # (625,)
    high_light = get_feature_map_img(out_pca, feature_map_size)
    return high_light


if __name__ == "__main__":
    # 适用于已保存CNN 提取的feature map
    mask_folder = './temporary_folder/rough_mask'
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    
    IMAGE_DIR = './temporary_folder/test_ori/'    # 测试图片文件夹
    feature_map_dir = './temporary_folder/feature_maps/' # 测试图片的feature map文件夹
    feature_map_size = 25
    for path in os.listdir(feature_map_dir):
        if 'npy' not in path:
            continue
        feature_map = get_feat_map(feature_map_dir, path, feature_map_size)
        rough_mask = colocalization(feature_map, feature_map_size)  # 生rough mask
        rough_mask_path = os.path.join(mask_folder, (path.split('.')[0] + '_roughmask.png'))
        rough_mask.save(rough_mask_path)
        mask = np.array(rough_mask)
        print(mask.shape)
    
    