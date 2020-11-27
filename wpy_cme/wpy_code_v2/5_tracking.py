# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image 
import copy 

import datetime
import util

IMAGE_SIZE = util.IMAGE_SIZE
MIN_MARGIN = 10
MIN_CME_WIDTH = 10

def img_polar_transform(input_img, center = (IMAGE_SIZE/2,IMAGE_SIZE/2), r_range=(0,int(IMAGE_SIZE/2*1.414)), theta_range=(0,360), r_step=1.0, theta_step=360.0/(360.0)):
    minr,maxr = r_range
    mintheta,maxtheta = theta_range
    H = int((maxr-minr)/r_step+1)#
    W = int((maxtheta-mintheta)/theta_step+1)#361

    output_img = np.zeros((H,W),input_img.dtype)
    x_center,y_center=center

    r = np.linspace(minr,maxr,H)
    r = np.tile(r,(W,1))
    r = np.transpose(r)
    theta = np.linspace(mintheta,maxtheta,W)
    theta = np.tile(theta,(H,1))
    x,y = cv2.polarToCart(r,theta,angleInDegrees=True)
	
    for i in range(H):
        for j in range(W):
            px = int(round(x[i,j]) + x_center)
            py = int(round(y[i,j]) + y_center)
			#最后10
            if ((px>=0 and px<=IMAGE_SIZE-1) and (py>=0 and py<=IMAGE_SIZE-1)):
                output_img[i,j]=input_img[px,py]
    return output_img
	
def get_cme_img(input_img,cme_list,theta_rangge=(0,360),r_step=1.0,theta_step=360.0/(360.0)):
    #保留cme部分，去掉非cme部分
    h,w = (IMAGE_SIZE,IMAGE_SIZE)
    center = (int(h/2),int(w/2))
    x_center,y_center=center
    r_range = (0,int(IMAGE_SIZE/2*1.414))
    
    def if_cme(theta):
    	for cme in cme_list:
    		if cme[0]<0:
    			if int(theta[0,0]/theta_step)>=(360.+cme[0]) or int(theta[0,0]/theta_step)<=cme[1]:
    				return True
    		else:
    			if int(theta[0,0]/theta_step)>=cme[0] and int(theta[0,0]/theta_step)<=cme[1]:
    				return True
    	return False
    
    for x in range(IMAGE_SIZE):
    	for y in range(IMAGE_SIZE):
    		px = x - x_center
    		py = y - y_center
    		r,theta=cv2.cartToPolar(px,py,angleInDegrees=True)
    		if if_cme(theta)==False:
    			input_img[x,y] = 0.0
    return input_img
	
def get_polar_images(now_dir, name):
    image = cv2.imread(now_dir + name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	
    polar_image = img_polar_transform(image[:,:IMAGE_SIZE])
    polar_cme = img_polar_transform(image[:,IMAGE_SIZE:])
	
    return polar_image, polar_cme

	
def compute_single_polar_max_height(img):
    h, t = img.shape
	#print h,t
    length = np.zeros((t)) + 2.
    for theta in range(t):
        w = np.where(img[:, theta]>0.5)

        if len(w[0])>2:
            length[theta] = w[0][-1] / float(h) * 6.2*1.414
    return length
	
def compute_polar_max_height(img_list):
    out_list = []
    for i in img_list:
        out_list.append(compute_single_polar_max_height(i))
    return np.asarray(out_list)
	
def refine_cme(cme_list):
	#根据相邻两个cme的间隔（间隔小于10）继续进行合并

    if len(cme_list)==0:
        return []
    for now_index in range(len(cme_list)-2,0,-1):
        #比较后一个cme的起始位置，如果与后一个cme的位置过近，则合并
        if abs(cme_list[now_index][1] - cme_list[now_index+1][0]) < MIN_MARGIN:
            cme_list[now_index][1] = cme_list[now_index+1][1]
            cme_list.remove(cme_list[now_index+1])

    #是否合并第一个与最后一个
    if (cme_list[0][0] - cme_list[-1][1] + 360)%360 <MIN_MARGIN:
        if (cme_list[0][1] + 360 - cme_list[-1][0]) > MIN_CME_WIDTH:		    
            cme_list[0][0] = cme_list[-1][0] - 360
            cme_list = cme_list[:-1]

    #删除所有小于MIN_CME_WIDTH的事件
    for now_index in range(len(cme_list)-2,0,-1):
        if (cme_list[now_index][1] - cme_list[now_index][0]) < MIN_CME_WIDTH:
            cme_list.remove(cme_list[now_index])

    return cme_list
    
def combine_cme_region(polar_cme_height_list):
    MIN_WIDTH = 2
    temp = np.sum(np.int32(polar_cme_height_list>4.0), axis=0)# * np.sum(np.int32(polar_cme_height_list<3.5), axis = 0)
    cme_theta = np.int32(temp>0)
	
    #combine_cme_region	
    a_start = 0
    a_begin_flag = True
    a_end = 0
    cme_list = []
    for i, flag in enumerate(cme_theta):
        if flag == 1:
            a_end = i
            if a_begin_flag:
                a_start = i
                a_begin_flag = False
            if i == len(cme_theta)-1:
                if a_end - a_start>MIN_WIDTH:
                    cme_list.append([a_start, a_end])
        else:
            if not a_begin_flag:
                if a_end - a_start>MIN_WIDTH:
                    cme_list.append([a_start, a_end])
                a_begin_flag = True
    return refine_cme(cme_list)
	
def compute_velocity(max_, min_, time_):#一个太阳半径等于696300km
    return (max_-min_) * 696300. /time_ #km/s

def compute_height_time_velocity(cme, polar_height_list, polar_seconds_list):
    #计算当前cme的最大高度，速度等参数信息
    def get_median_max(l):
        l_arg = np.argsort(l)#升序
        median_arg = len(l)//2
        return l_arg[-1], l[l_arg[-1]], l_arg[median_arg], l[l_arg[median_arg]]
		
    v_list = []
    v_param_list = []
    polar_height_list = np.asarray(polar_height_list)

    for angle_index in range(cme[0], cme[1]+1):
        _now = (angle_index + 360)%360
        idx_max = np.argmax(polar_height_list[:, _now])
        idx_min = np.argmax(-polar_height_list[:, _now])

        time_passed = polar_seconds_list[idx_max] - polar_seconds_list[idx_min]
        v = compute_velocity(polar_height_list[idx_max, _now], polar_height_list[idx_min, _now], time_passed)
        v_list.append(v)
        v_param_list.append([angle_index, polar_height_list[idx_min:idx_max+1, _now], idx_min, idx_max])
    max_index, max_v, median_index, median_v = get_median_max(np.asarray(v_list))
	
    return median_v, max_v, v_param_list[median_index][0], v_param_list[median_index][1], v_param_list[median_index][2], v_param_list[median_index][3]
	
#处理一组输入
def compute_one(input_id, cme_id_base):
    now_dir = './out_4/'+str(input_id)+'/'

    if not os.path.exists(now_dir):
        return cme_id_base
    
    file_list = os.listdir(now_dir)
	
    polar_image_list = []
    polar_cme_list = []

    polar_seconds_list = []
    image_name_list = []
    start_time_flag = True
    #每张图像加载并转换成极坐标
    for name in file_list:
        if '_flow' in name:
            continue
        if start_time_flag:
            start_data = datetime.datetime.strptime(name[:-4],"%Y-%m-%dT%H_%M_%S")
            start_time_flag = False
            polar_seconds_list.append(0)
        else:
            now_data = datetime.datetime.strptime(name[:-4],"%Y-%m-%dT%H_%M_%S")
            delta = (now_data - start_data)
            if delta.days < 0:
                delta = delta.seconds - 24*3600
            else:
                delta = delta.seconds
            polar_seconds_list.append(delta)
        image_name_list.append(name)
        polar_image, polar_cme = get_polar_images(now_dir, name)
        polar_image_list.append(polar_image)
        polar_cme_list.append(polar_cme)
        #写极坐标图像，调试用
        #all_images = np.vstack((polar_image,polar_cme))
        #cv2.imwrite(now_dir+name+'_flow.png', all_images)
    print('input_id %d read images.......'%(input_id), end="")
    #计算最大高度,合并区间
    polar_cme_height_list = compute_polar_max_height(polar_cme_list)
    refined_cme_list = combine_cme_region(polar_cme_height_list)
    
    output_cme_num = 0	
    for cme in refined_cme_list:
        median_vel, max_vel, max_vel_PA, max_vel_PA_height_list, start_idx, end_idx = compute_height_time_velocity(cme, polar_cme_height_list, polar_seconds_list)
        central_angle = ((cme[1] + cme[0])/2 + 360)%360
        angular_width = (cme[1] - cme[0])  
        if angular_width > 320:
            angular_width = 360
        #条件1：是否存在超过2帧
        if end_idx - start_idx <= 2:
            continue
        if angular_width < MIN_CME_WIDTH:
            continue
        #写文件  
        #条件2：判断是否到达视口边界，视口边界是6，但是由于分割的效果并不会不好，这里写6会丢失很多事件
        if max_vel_PA_height_list[-1]<4.2:
            continue		
        out_dir = './out_5/'+str(cme_id_base+output_cme_num)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)		
        for img_id, name in enumerate(image_name_list[start_idx:end_idx+1]):

            now = cv2.imread(now_dir+name)[:,IMAGE_SIZE:]
            orgin = cv2.imread(now_dir+name)[:,:IMAGE_SIZE]
            new_red = get_cme_img(now, [cme])
            new_red = cv2.split(new_red)[2]
            r  =Image.fromarray(new_red).convert('L')
            g = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('L')
            b = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('L')
            a = Image.fromarray(new_red/float(IMAGE_SIZE)*70.).convert('L')
            high_light=Image.merge("RGBA", (r, g, b, a))
            img = Image.fromarray(orgin).convert('L')
            img = img.convert('RGBA')
            img.paste(high_light, (0,0), mask=high_light)
            img.save(out_dir+'/'+name.split('.')[0]+'_cme.png')
            
        w = open(out_dir+'/'+'cme_params.txt','w')	
        w.write(str(cme[0])+' '+str(cme[1])+'\n')
        for i in max_vel_PA_height_list:
            w.write(str(i)+' ')
        w.write('\n'+image_name_list[start_idx]+' '+image_name_list[end_idx]+'\n')
        w.write('\n')
        w.write('Central PA : %f\n'%(central_angle))
        w.write('Angular Width : %f\n'%(angular_width))
        w.write('Median Linear Velocity : %f\n'%(median_vel))
        w.write('Max Linear Velocity : %f\n'%(max_vel))
        w.close()
        output_cme_num += 1
    print('.....tracking %d cmes done!'%(output_cme_num))
    return output_cme_num + cme_id_base
		
if __name__ == '__main__':

    OUT_IMAGE_DIR = './out_5/'

    if not os.path.exists(OUT_IMAGE_DIR):
	    os.makedirs(OUT_IMAGE_DIR) 
		
    num = len(os.listdir('./out_4'))
    cme_id_base = 1
    for i in range(1,num+1):
        cme_id_base = compute_one(i, cme_id_base)
