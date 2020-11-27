import os
import cv2
import numpy as np
from PIL import Image
import copy
from sklearn import linear_model
import datetime
def img_polar_transform(input_img,center,r_range,theta_rangge=(0,360),r_step=0.5,theta_step=360.0/(360.0)):
    print("theta_step",theta_step)
    minr,maxr=r_range
    mintheta,maxtheta=theta_rangge
    H=int((maxr-minr)/r_step+1)
    W=int((maxtheta-mintheta)/theta_step+1)
    # print(H,W,theta_step)
    print(input_img.dtype)
    output_img=np.zeros((H,W),input_img.dtype)
    x_center,y_center=center

    r=np.linspace(minr,maxr,H)
    r=np.tile(r,(W,1))
    r=np.transpose(r)
    theta=np.linspace(mintheta,maxtheta,W)
    theta=np.tile(theta,(H,1))
    x,y=cv2.polarToCart(r,theta,angleInDegrees=True)

    for i in range(H):
        for j in range(W):
            px=int(round(x[i,j])+x_center)
            py = int(round(y[i, j]) + y_center)
            if ((px>=5 and px<=255-5) and (py>=5 and py<=255-5)):
            	#print(px,py,input_img[px,py])
                output_img[i,j]=input_img[px,py]
    return output_img

def denoise(input_img,cme_list,theta_rangge=(0,360),r_step=0.5,theta_step=360.0/(360.0)):
    h,w = (256,256)
    x_center,  y_center = (int(h/2),int(w/2))
    r_range = (0,int(127*1.414))

    def if_cme(theta):
    	for cme in cme_list:
    		if cme[0]<0:
    			if int(theta[0,0]/theta_step)>=(360.+cme[0]) or int(theta[0,0]/theta_step)<=cme[1]:
    				return True
    		else:
    			if int(theta[0,0]/theta_step)>=cme[0] and int(theta[0,0]/theta_step)<=cme[1]:
    				return True
    	return False

    for x in range(256):
    	for y in range(256):
    		px = x - x_center
    		py = y - y_center
    		r,theta=cv2.cartToPolar(px,py,angleInDegrees=True)

    		if if_cme(theta)==False:
    			input_img[x,y] = 0.0
    return input_img

def refine_cme(cme_list):
	def compute_length_l(l, max_len = 10):
		out = []
		for i in range(len(l)):
			out.append(l[i][1]-l[i][0])
		return np.array(out)

	len_list = compute_length_l(cme_list)
	if len(len_list)==0:
		return []
	while (np.min(len_list)<10):
		need_del = np.argmax(-len_list)
		#print need_del, cme_list
		if abs(cme_list[need_del][0] - cme_list[(need_del-1)%len(cme_list)][1])<10:
			cme_list[(need_del-1)%len(cme_list)][1] = cme_list[need_del][1]
			cme_list.remove(cme_list[need_del])
			len_list = compute_length_l(cme_list)
			if len(len_list)==0:
				return []
			continue
		if abs(cme_list[need_del][1] - cme_list[(need_del+1)%len(cme_list)][0])<10:
			cme_list[(need_del+1)%len(cme_list)][0] = cme_list[need_del][0]
			cme_list.remove(cme_list[need_del])
			len_list = compute_length_l(cme_list)
			if len(len_list)==0:
				return []
			continue
		cme_list.remove(cme_list[need_del])
		len_list = compute_length_l(cme_list)
		if len(len_list)==0:
			return []
	loop_flag = True
	while loop_flag:
		le_ = []
		for i in range(len(cme_list)-1):
			le_.append(abs(cme_list[i][1] - cme_list[(i+1)%len(cme_list)][0])%360)
		if np.sum(np.int32(np.array(le_)<10)) == 0:
			loop_flag = False
		temp_list = [cme_list[0]]
		for i in range(len(cme_list)-1):
			if np.array(le_)[i]>13:
				temp_list.append(cme_list[i+1])
			else:
				temp_list[-1][1] = cme_list[i+1][1]
		cme_list = copy.deepcopy(temp_list)

	if (cme_list[0][0] - cme_list[-1][1] + 360)%360 <10:

		cme_list[0][0] = cme_list[-1][0] - 360
		cme_list = cme_list[:-1]
		print('important',cme_list)

	return cme_list


def compute_length(img):
	h, t = img.shape
	#print h,t
	length = np.zeros((t)) + 90
	for theta in range(t):
		w = np.where(img[:, theta]>0)
		#print w
		if len(w[0])>2:
			length[theta] = w[0][-3]

	return length

def compute_one(id, ratio,cme_id_base):
	#data = '20140212/'
	max_length_solar_rate = 6.0/(255.*1.414)
	now_dir = './out_4/'+str(id)+'/'
	if not os.path.exists(now_dir):
		return
	file_list = os.listdir(now_dir)
	polar_seconds_list = []
	polar_list = []
	polar_im_list = []
	polar_length = []

	polar_name = []
	start_time_flag = True
	for name in file_list:
		if '_flow' in name or '_cme' in name:
			continue
		if start_time_flag:
			start_data = datetime.datetime.strptime(name[:-4],"%Y-%m-%dT%H_%M_%S")
			start_time_flag = False
			polar_seconds_list.append(0)
		else:
			now_data = datetime.datetime.strptime(name[:-4],"%Y-%m-%dT%H_%M_%S")
			delta = (now_data - start_data)
			if delta.days < 0 :
				delta = delta.seconds - 24*3600
			else:
				delta = delta.seconds
			polar_seconds_list.append(delta)
		polar_name.append(name)
		h,w = (256,256)
		center = (int(h/2),int(w/2))
		r_range = (0,int(127*1.414))

		now = cv2.imread(now_dir+name)[:,256:]
		now = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
		now_polar = img_polar_transform(now,center,r_range)

		now_im = cv2.imread(now_dir+name)[:,:256]
		now_im = cv2.cvtColor(now_im, cv2.COLOR_BGR2GRAY)
		now_polar_im = img_polar_transform(now_im,center,r_range)
		polar_im_list.append(now_polar_im)
		polar_list.append(now_polar)

		polar_length.append(compute_length(now_polar))

	polar_list = np.mean(polar_list, axis = 0)
	length = compute_length(polar_list)



	mean_length = np.mean(length)
	mask = np.int32(length > mean_length*ratio)
	h, T = polar_list.shape
	a_start = 0
	a_begin_flag = True
	a_end = 0
	cme_list = []
	for i, flag in enumerate(mask):
		if flag == 1:

			a_end = i
			if a_begin_flag:
				a_start = i
				a_begin_flag=False

			if i == len(mask)-1:
				if a_end - a_start>8:
					cme_list.append([a_start, a_end])

		else:

			if not a_begin_flag:
				if a_end - a_start>8:
					cme_list.append([a_start, a_end])
				a_begin_flag = True

	new_mask = np.zeros((T))

	cme_list_new = refine_cme(cme_list)
	#print cme_list_new

	out_dir = './out_5/'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	polar_length = np.array(polar_length)

	max_min_length_cme = []# max min !

	cme_list_final = []
	for cme in cme_list_new:
		if cme[0]<0:
			now_ = np.mean(np.concatenate([polar_length[:,0:cme[1]], polar_length[:,360+cme[1]:]], axis = 1), axis = 1)
		else:
			now_ = np.mean(polar_length[:,cme[0]:cme[1]], axis = 1)
		now_max = np.argmax(now_)
		i_tmp=0
		while now_max == 0:
			i_tmp+=1
			now_max = np.argmax(now_[i_tmp:])+i_tmp
		if len(now_[:now_max]) == 0:
			continue
		now_min = np.argmax(-now_[:now_max])
		print(now_min, now_max)
		if now_[now_max]>170 and (now_[now_max] - now_[now_min])>35:
			cme_list_final.append(cme)
			linear_vel = (now_[now_max] - 90.) / (polar_seconds_list[now_max] - polar_seconds_list[now_min])
			max_min_length_cme.append([now_min, now_max, now_[now_max], now_[now_min], linear_vel])
			if cme[0] < 0:
				for j in range(0,cme[1]+1,1):
					new_mask[j] = 1.
				for j in range(cme[0]+360,361,1):
					new_mask[j] = 1.
			else:
				for j in range(cme[0],cme[1]+1,1):
					new_mask[j] = 1.

	for i in range(T):
		if new_mask[i] == 0:
			polar_list[:, i] = 0.

	polar_im_list = np.mean(polar_im_list, axis = 0)
	print(max_min_length_cme)
	for cme_id, cme in enumerate(cme_list_final):

		#regr = linear_model.LinearRegression()
		#regr.fit(np.array(polar_seconds_list).reshape(-1, 1), max_length_solar_rate*for_linear_reg.reshape(-1, 1))
		#print for_linear_reg, regr.coef_[0][0]*696300., regr.intercept_[0]
		#param_a, param_b = regr.coef_[0][0], regr.intercept_[0]
		now_cme_dir = str(cme_id + cme_id_base)
		if not os.path.exists(out_dir+now_cme_dir):
			os.makedirs(out_dir+now_cme_dir)

		_start = max_min_length_cme[cme_id][0]
		if _start >= 1:
			_start -= 1
		_end = max_min_length_cme[cme_id][1]
		if _end<len(polar_name)-1:
			_end += 1
		for img_id,name in enumerate(polar_name[_start:_end+1]):
			h,w = (256,256)
			center = (int(h/2),int(w/2))
			r_range = (0,int(127*1.414))
			now = cv2.imread(now_dir+name)[:,256:]
			orgin = cv2.imread(now_dir+name)[:,:256]
			new_red = denoise(now, [cme])
			new_red = cv2.split(new_red)[2]
			r  =Image.fromarray(new_red).convert('L')
			g = Image.fromarray(np.zeros((256,256))).convert('L')
			b = Image.fromarray(np.zeros((256,256))).convert('L')
			a = Image.fromarray(new_red/255.*70).convert('L')
			high_light=Image.merge("RGBA", (r, g, b, a))
			img = Image.fromarray(orgin).convert('L')
			img = img.convert('RGBA')
			img.paste(high_light, (0,0), mask=high_light)
			img.save(out_dir+now_cme_dir+'/'+name.split('.')[0]+'_cme.png')
			#print(name)
		w = open(out_dir+now_cme_dir+'/'+'cme_params.txt','w')
		#all_images = np.vstack((polar_list,polar_im_list))
		w.write(str(cme[0])+' '+str(cme[1])+'\n')
		w.write(str(max_min_length_cme[cme_id][2])+' '+str(max_min_length_cme[cme_id][3])+'\n')
		w.write(polar_name[_start]+' '+polar_name[_end]+'\n')
		#w.write(str(param_a)+' '+str(param_b)+'\n')
		w.write('\n')
		w.write('Central PA : %f\n'%(((cme[0]+cme[1])/2.+180.)%360))
		w.write('Angular Width : %f\n'%(abs(cme[0]-cme[1])))
		w.write('Linear Velocity : %f\n'%(max_min_length_cme[cme_id][-1]*max_length_solar_rate*696300.))
		w.close()
		#cv2.imwrite(now_dir+'_flow_new.png', all_images)
	return len(cme_list_final) + cme_id_base
	'''
	flow = cv2.calcOpticalFlowFarneback(pred_polar, next_polar, None, 0.5, 3, 15, 3, 7, 1.5, 0)

		
	all_images = np.vstack((next_polar, flow[..., 0], flow[..., 1]))
		
	cv2.imwrite(input_images_list[i]+'_flow.png', all_images)
	'''
if __name__ == '__main__':
	num = len(os.listdir('out_4'))
	cme_id_base = 0
	for i in range(1,num+1):
		cme_id_base = compute_one(i,0.95,cme_id_base)
