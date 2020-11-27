'''
Author       : ZHP
Date         : 2020-11-23 11:32:20
LastEditors  : ZHP
LastEditTime : 2020-11-27 16:47:55
FilePath     : /Earlier_Project/rewrite/refinement.py
Description  : 对Wpy学长的refinement代码进行整合，改动少量代码，添加个人注释
Copyright 2020 ZHP
'''
import cv2
from PIL import Image  
import os
import numpy as np
import maxflow

Use_Graphcut=True
IMAGE_SIZE = 512

def img_polar_transform(input_img,center,r_range,theta_rangge=(0,360),r_step=0.5,theta_step=360.0/(90.)):
    '''
    description: 极坐标转换
    param {
        input_img : 输入feature map
        center : 图片中心坐标
        r_range : 半径radius范围
        theta_rangge : 角度范围
        r_step : 半径步长
        theta_step : 角度步长
    }
    return {
        转换后图片
    }
    '''
    minr,maxr=r_range  # (0, 200)
    mintheta,maxtheta=theta_rangge # (0, 360)
    H=int((maxr-minr)/r_step+1) # 401
    W=int((maxtheta-mintheta)/theta_step+1) # 91
    output_img=np.zeros((H,W),input_img.dtype)
    x_center,y_center=center # (256, 256)
    
    r=np.linspace(minr,maxr,H) # (401,)  0~200 间隔为0.5: 0. ,0.5,1. ,1.5,..,200.  代表极坐标的p,半径
    r=np.tile(r,(W,1))    # 复制 r-->(91, 401)
    r=np.transpose(r)    # (401, 91) ,每列相同
    theta=np.linspace(mintheta,maxtheta,W)  # (91,)  0~360 间隔为4：0,4,8,..,360.  代表极坐标的角度：theta
    theta=np.tile(theta,(H,1)) # (401, 91)，每行相同

    x,y=cv2.polarToCart(r,theta,angleInDegrees=True)  # 极坐标转为笛卡尔坐标系   x:笛卡尔坐标系x轴值，y:y轴值
    # x.shape, y.shape   # ((401, 91), (401, 91))

    x.min(), x.max(), y.min(), y.max()   # (-200.0, 200.0, -199.878191947937, 199.878191947937)
    # 获取以(x_center, y_center)为圆心的极坐标对应的笛卡尔坐标系
    for i in range(H):
        for j in range(W):
            px = int(round(x[i,j])+x_center)
            py = int(round(y[i, j]) + y_center)
            if ((px>=0 and px<(2*x_center-1) and (py>=0 and py<(2*y_center-1)))):
                # print(px,py)
                output_img[i,j]=input_img[px,py]

    return output_img


def compute_max_length(img):
    '''
    返回img每列中大于10的数中最大的行索引
    '''
    num_theta = img.shape[1]  # 91
    out_all = np.ones(img.shape[1])*(img.shape[0]-1)
    # out_all.shape, out_all[3]  # (91,), 400.0
    for i in range(num_theta):
        index = np.where(img[:,i]>10)   # 第i列大于10的相对索引
        try:
            out_all[i] = np.max(index)  # 每列中大于10的数中最大的行索引

        except:
            pass
    return out_all


def create_graph_and_cut(image, mask):
    '''
    description: 生成最小割
    param {
        image : 原图
        mask : 极坐标转换后的mask
    }
    return {
        图割后的mask
    }
    '''
    g = maxflow.Graph[float]()   # 创建最大流图
    nodes = g.add_grid_nodes(image.shape[:2])  # 添加节点（H,W）
    bottom_weights = np.zeros(image.shape[:2])  # 初始权重
    left_weighta = np.zeros(image.shape[:2])

    # 0~H-1行 依次减去 1~H行
    bottom_weights = 100. / np.exp(((image[:-1,:] - image[1:,:]) ** 2) / (2 * 1 ** 2))   # (511, 512)

    # 0~W-1列依次减去 1~W列
    left_weights = 100. /np.exp(((image[:,:-1] - image[:,1:]) ** 2) / (2 * 1 ** 2))  # (512, 511)

    bottom_weights = np.vstack((bottom_weights, np.zeros(image.shape[1])))  # 在最后添加一行0，shape成(512, 512)
    left_weights = np.hstack((left_weights, np.zeros((image.shape[0], 1)))) # 在最后添加一列0，shape成(512, 512)
    bottom_structure = np.array([[0, 0, 0],
                                [0, 0, 0],
                                [0, 1, 0]])

    left_structure = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [0, 0, 0]])

    g.add_grid_edges(nodes, bottom_weights, structure=bottom_structure, symmetric=True)
    g.add_grid_edges(nodes, left_weights, structure=left_structure, symmetric=True)

    g.add_grid_tedges(nodes, (1-mask)*5., (mask)+image*0.1)
    curr_maxflow = g.maxflow()
    return g.get_grid_segments(nodes)   # 返回ndarray


def create_graph_and_cut_refine(image, mask):
	g = maxflow.Graph[float]()
	nodes = g.add_grid_nodes(image.shape[:2])
	bottom_weights = np.zeros(image.shape[:2])
	left_weighta = np.zeros(image.shape[:2])
   
	bottom_weights = 2. / np.exp(((image[:-1,:] - image[1:,:]) ** 2) / (2 * 10. ** 2))
	left_weights = 2. /np.exp(((image[:,:-1] - image[:,1:]) ** 2) / (2 * 10. ** 2))
	bottom_weights = np.vstack((bottom_weights, np.zeros(image.shape[1])))
	left_weights = np.hstack((left_weights, np.zeros((image.shape[0], 1))))

	bottom_structure = np.array([[0, 0, 0],
								 [0, 0, 0],
								 [0, 1, 0]])
	left_structure = np.array([[0, 0, 0],
							   [1, 0, 0],
							   [1, 0, 0]])
	g.add_grid_edges(nodes, bottom_weights, structure=bottom_structure, symmetric=True)
	g.add_grid_edges(nodes, left_weights, structure=left_structure, symmetric=True)
	g.add_grid_tedges(nodes, (1-mask)*1.8, mask*0.7+image*0.08)
	curr_maxflow = g.maxflow()
	return g.get_grid_segments(nodes)


def denoise(mask_image, max_mask, center,r_range,theta_rangge=(0,360),r_step=0.5,theta_step=360.0/(90.)):
    minr, maxr = r_range
    mintheta, maxtheta = theta_rangge
    H = int((maxr-minr)/r_step+1)
    W = int((maxtheta-mintheta)/theta_step+1)
    x_center, y_center = center

    for x in range(512):
        for y in range(512):
            px = x - x_center
            py = y - y_center
            r,theta = cv2.cartToPolar(px,py,angleInDegrees=True)  # 转极坐标
            now_r = max_mask[int(theta[0,0]/theta_step)]*r_step
            if r[0,0] > now_r or now_r < 30:
                mask_image[x,y] = 0.
    return mask_image


def get_refine_mask_folder(outImageDir, inputPolarImageDir):
    '''
    description: 对整个文件夹的rough mask生成refine后的mask
    param {
        outImageDir : 生成的mask和原图拼接图片保存位置
        inputPolarImageDir : 输入的初始co-locolization后的图 
    }
    '''
    if not os.path.exists(outImageDir):
        os.makedirs(outImageDir) 
    file_list = os.listdir(inputPolarImageDir)
    for file_idx in range(len(file_list)):
        polar_image_name = file_list[file_idx]
        if 'npy' in polar_image_name or 'flow' in polar_image_name:
            continue
        concat_image = cv2.imread(inputPolarImageDir + polar_image_name)
        concat_image= cv2.split(concat_image)[2]   # 拆分通道，提取R通道
        origin_image_save = concat_image[:,0:IMAGE_SIZE]  # 提取原图
        # 小于均值的设为0，其余减均值并控制在<255
        origin_image = np.uint8(np.clip(origin_image_save - np.mean(origin_image_save), 0., 255.))

        
        mask = concat_image[:,512:]  # 提取mask
        h,w = origin_image.shape[:2]
        center = (int(h/2),int(w/2))
        r_range = (0,200)
        mask_polar = img_polar_transform(mask, center,r_range)
        origin_max_length = compute_max_length(mask_polar)

        if Use_Graphcut:
            mask = np.float32(mask)/255.

            sgm = create_graph_and_cut(origin_image, mask)

            mask2 = np.uint8(np.logical_not(sgm)==0)
            mask2 = denoise(mask2, origin_max_length, center=center, r_range=r_range)
            sgm_refine = create_graph_and_cut_refine(origin_image, mask2)
            mask3 = np.uint8(np.logical_not(sgm_refine)==0)
            
            Zconcat_target = Image.new('RGB', (IMAGE_SIZE*2, IMAGE_SIZE))
            r  =Image.fromarray(mask3*255).convert('L')
            g = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('L')
            b = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('L')
            high_light=Image.merge("RGB", (r, g, b))
            origin_image = Image.fromarray(origin_image_save).convert('L')
            origin_image = origin_image.convert('RGB')
            
            Zconcat_target.paste(high_light, (IMAGE_SIZE,0))
            Zconcat_target.paste(origin_image, (0,0))
            save_path = OUT_IMAGE_DIR + polar_image_name
            Zconcat_target.save(save_path)
            print('{} mask 生成成功！保存在 {}'.format(polar_image_name, save_path))


def get_refine_mask(image, mask):
    '''
    description: 对单张rough mask生成refine mask
    param {
        image : 原图
        mask : co-localization 后的mask,ndarray,BGR
    }
    return {
        high_light : 最终的mask, ndarray,np.uint8
        Zconcat_target : 原图和mask拼接起来 ndarray,np.uint8

    }
    '''
    mask = cv2.split(mask)[2]   # 拆分通道，提取R通道

    # 小于均值的设为0，其余减均值并控制在<255
    origin_image = np.uint8(np.clip(image - np.mean(image), 0., 255.))
    h,w = mask.shape[:2]
    center = (int(h/2),int(w/2))
    r_range = (0,200)
    
    mask_polar = img_polar_transform(mask, center,r_range)  # 坐标转换
    origin_max_length = compute_max_length(mask_polar)

    if Use_Graphcut:
        mask = np.float32(mask)/255.
        sgm = create_graph_and_cut(origin_image, mask)  # 第一次图割
        mask2 = np.uint8(np.logical_not(sgm)==0)
        mask2 = denoise(mask2, origin_max_length, center=center, r_range=r_range)   # 这一步作用不大，可注释掉
        sgm_refine = create_graph_and_cut_refine(origin_image, mask2)  # 第二次图割
        mask3 = np.uint8(np.logical_not(sgm_refine)==0)

        high_light = np.zeros((h, w, 3))
        high_light[:,:,0] = mask3 * 255
        origin_image = np.expand_dims(image, 2).repeat(3, axis=2)
        Zconcat_target = np.concatenate([origin_image, high_light], axis=1)
        return np.uint8(high_light), np.uint8(Zconcat_target)



if __name__ == '__main__':
    # 对rough mask通过图割进行refine，得到最终mask
    origin_image_dir = 'temporary_folder/test_ori/'
    rough_mask_dir = 'temporary_folder/rough_mask/'
    refine_mask_dir = 'temporary_folder/refine_mask/'

    now_path = os.getcwd()   # 防止opencv因为路径问题读取错误
    
    if not os.path.exists(refine_mask_dir):
        os.makedirs(refine_mask_dir)
    for img in os.listdir(rough_mask_dir):
        if '_roughmask.png' not in img:
            continue
        rough_mask = cv2.imread(os.path.join(now_path, rough_mask_dir, img))  # rough mask
        origin = cv2.imread(os.path.join(now_path, origin_image_dir, (img.split('_roughmask')[0] + '.png')), 0)  # 原图
        result, cat_result = get_refine_mask(origin, rough_mask)  # 唯一调用函数

        # save
        Image.fromarray(result).save(os.path.join(now_path, refine_mask_dir, (img.split('_roughmask')[0] + '_mask.png')))
        Image.fromarray(cat_result).save(os.path.join(now_path, refine_mask_dir, (img.split('_roughmask')[0] + '_mask_cat.png')))
    