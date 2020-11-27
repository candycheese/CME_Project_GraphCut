import os
import cv2
import numpy as np
import json
import util
import shutil

if __name__ == '__main__':
        # input_images_list = open(util.LIST_FILE,'r').read().split()
        # input_images_list = json.load(open(util.LIST_FILE))

        input_images_dir = './out_3/'
        input_images_list_real = os.listdir(input_images_dir)
        # print(input_images_list_real)
        # for idx in range(len(input_images_list)):
        #         input_images_list[idx] = input_images_list[idx].replace(':','_')
        input_images_label_table = json.load(open(util.TEST_LABEL_FILE))

        out_dir = './out_4/'
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)
		
        cme_id = 0
        now_cme_list = []
        need_new_cme_flag = True
        # print(input_images_list_real)
        # input_images_label_table.sort()
        input_images_list=[]
        for i in input_images_label_table:
                input_images_list.append(i)
                # print(input_images_list(0))
        input_images_list.sort()
        # print(input_images_list)

        for i in range(0, len(input_images_list)-3):
                # print(input_images_list[i],'----')
                if not input_images_list[i]  in input_images_list_real:
                        print("continue")
                        continue

                # print(input_images_list[i-1], input_images_list[i])
		
                now_label = input_images_label_table[input_images_list[i] ]

                if now_label == 1:

                        if need_new_cme_flag:
                                now_cme_list = []
                        #if (i>=3) and (not i-3 in now_cme_list) and (input_images_list[i-3]  in input_images_list_real):
                                #now_cme_list.append(i-3)
                        #if (i>=2) and (not i-2 in now_cme_list) and (input_images_list[i-2]  in input_images_list_real):
                                #now_cme_list.append(i-2)
                        #if (i>=1) and (not i-1 in now_cme_list) and (input_images_list[i-1]  in input_images_list_real):
                                #now_cme_list.append(i-1)				
                        if (not i in now_cme_list):
                                now_cme_list.append(i)
                        if need_new_cme_flag and len(now_cme_list)>0:
                                need_new_cme_flag = False
                                cme_id += 1
 
                if now_label == 0 and need_new_cme_flag == False:
                        if not (input_images_list[i+1]  in input_images_list_real):
                                continue
                        if input_images_label_table[input_images_list[i+1] ] == 1:
                                now_cme_list.append(i)
                                continue
				
                        if (input_images_list[i+1]  in input_images_list_real):
                                now_cme_list.append(i+1)
                        #if (input_images_list[i+2]  in input_images_list_real):
                                #now_cme_list.append(i+2)
                        #if (input_images_list[i+3]  in input_images_list_real):
                                #now_cme_list.append(i+3)
                        #if (input_images_list[i+4]  in input_images_list_real):
                                #now_cme_list.append(i+4)
                        #if (input_images_list[i+5]  in input_images_list_real):
                                #now_cme_list.append(i+5)
                        need_new_cme_flag = True
                        # print(now_cme_list)
                if len(now_cme_list)<4:
                        continue				
                if need_new_cme_flag or i==len(input_images_list)-3:
                        if not os.path.exists(out_dir+str(cme_id)+'/'):
                                os.makedirs(out_dir+str(cme_id)+'/')
			
                        for j in now_cme_list:
			
                                oldname = input_images_dir+input_images_list[j] 
                                newname = out_dir+str(cme_id)+'/'+input_images_list[j] 
				
                                shutil.copyfile(oldname, newname)
	
