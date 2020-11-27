import os
import cv2
import numpy as np
import json

import shutil

if __name__ == '__main__':
	#data = '20140212/'
	
	input_images_list = open('2011_12_train_list_file.txt','r').read().split()
	print("input_images_list",input_images_list)
	input_images_dir = './out_3/'
	input_images_list_real = os.listdir(input_images_dir)
	print("input_images_list_real",input_images_list_real)
	input_images_label_table = json.load(open('201112_labels.txt'))
	print("input_images_label_table", input_images_label_table)
	
	out_dir = './out_4/'
	
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
		
	cme_id = 0
	now_cme_list = []
	need_new_cme_flag = True
	for i in range(0, len(input_images_list)-3):
		if not input_images_list[i]+'.png' in input_images_list_real:
			continue

		#print(input_images_list[i-1], input_images_list[i])
		
		now_label = input_images_label_table[input_images_list[i]+'.png']
		print("now_label",now_label)
		#print now_label
		if now_label == 1:
			#print input_images_list[i]
			if need_new_cme_flag:
				now_cme_list = []
			if (i>=1) and (not i-1 in now_cme_list) and (input_images_list[i-1]+'.png' in input_images_list_real):
				now_cme_list.append(i-1)				
			if (not i in now_cme_list):
				now_cme_list.append(i)
			if need_new_cme_flag and len(now_cme_list)>0:
				need_new_cme_flag = False
				cme_id += 1
 
		if now_label == 0 and need_new_cme_flag == False:
			if not (input_images_list[i+1]+'.png' in input_images_list_real):
				continue
			if input_images_label_table[input_images_list[i+1]+'.png'] == 1:
				now_cme_list.append(i)
				continue
				
			if (input_images_list[i+1]+'.png' in input_images_list_real):
				now_cme_list.append(i+1)

			need_new_cme_flag = True
			print(now_cme_list)
			
		if need_new_cme_flag or i==len(input_images_list)-2:
			if not os.path.exists(out_dir+str(cme_id)+'/'):
				os.makedirs(out_dir+str(cme_id)+'/')
			
			for j in now_cme_list:
			
				oldname = input_images_dir+input_images_list[j]+'.png'
				newname = out_dir+str(cme_id)+'/'+input_images_list[j]+'.png'
				
				shutil.copyfile(oldname, newname)
	
