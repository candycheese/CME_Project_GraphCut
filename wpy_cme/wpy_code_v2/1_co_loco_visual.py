from sklearn.decomposition import PCA  
import cv2
from PIL import Image  
import os
import numpy as np
from skimage import measure,color
# import maxflow
import json
import time
import util
GLOBAL_FEATURE_MAP_SIZE =14  #25
# FEATURE_MAP_DIR = util.FEATURE_MAP_DIR#'./201201_out/test/'
FEATURE_MAP_DIR = '/disk/dataset/Earlier_Project/feature_maps/'
IMAGE_DIR = util.IMAGE_DIR#'./201201/'
use_graphcut=False
IMAGE_SIZE = util.IMAGE_SIZE
def get_feat_map(dir, picname):
    feat_map=np.load(dir+picname+'.npy')       #修改np.load为np.asarray  .txt为.npy  
    print(feat_map.shape)
    #     feat_map=np.asarray(open(dir+picname+'.txt', 'r').read().split(), dtype=np.float32)
    feat=feat_map.reshape(GLOBAL_FEATURE_MAP_SIZE * GLOBAL_FEATURE_MAP_SIZE, 512)#50)
#     feat_map = feat_map.reshape(-1, 512)
    print(feat.shape)
    return feat
	
def write_one_pca_feature_map(OUT_IMAGE_DIR, name, feature_map):
    Zconcat_target = Image.new('RGB', (IMAGE_SIZE*2, IMAGE_SIZE*2))
    zero=np.zeros((GLOBAL_FEATURE_MAP_SIZE, GLOBAL_FEATURE_MAP_SIZE),dtype = np.uint8)
    _max = 20
    _min = 0.
    
    # print('feature_map',feature_map)

#     print('feature_map %.4f'% feature_map)
    feature_map = feature_map*(feature_map>0.2)
    f = (feature_map - _min) / (_max-_min)

    w = open(OUT_IMAGE_DIR + name+'.txt', 'w')
	
    f = np.int32(f>0.0)*f
    for i in f:
        w.write(str(i)+'\n')
    w.close()
	
    r = Image.fromarray(f.reshape(GLOBAL_FEATURE_MAP_SIZE,GLOBAL_FEATURE_MAP_SIZE)*255).convert('L')
    g = Image.fromarray(zero).convert('L')
    b = Image.fromarray(zero).convert('L')
    #a = Image.fromarray(n.reshape(GLOBAL_FEATURE_MAP_SIZE,GLOBAL_FEATURE_MAP_SIZE)*70).convert('L')
    high_light=Image.merge("RGB", (r, g, b)) 	
    #origin = Image.open(IMAGE_DIR + name).convert('RGBA')
    #origin.paste(high_light, (0,0), mask=high_light) 
    #origin.save(OUT_IMAGE_DIR + name)
    high_light=high_light.resize((IMAGE_SIZE,IMAGE_SIZE))

    origin = Image.open(IMAGE_DIR + name).convert('RGB')
    Zconcat_target.paste(high_light, (IMAGE_SIZE,0))
    Zconcat_target.paste(origin, (0,0))
    Zconcat_target.save(OUT_IMAGE_DIR + name)

def define_trainer():
    #labels = open('test_result.txt','r')
    #labels_table = json.load(labels)
    #training PCA
    all_feature_map_list = []
    name_list = []
    pca_feature_dir = FEATURE_MAP_DIR
    for name in os.listdir(pca_feature_dir):
        #if labels_table[name[:-4]] == 0:
            #continue
        if 'graphcut' in name:
            continue
        try:
            all_feature_map_list.append(get_feat_map(pca_feature_dir, name.split('.')[0]+'.png'))
            name_list.append(name.split('.')[0]+'.png')
        except:
            continue
    N_images = len(name_list)
    print(N_images)
    feature_map_PCA = np.asarray(all_feature_map_list).reshape(N_images, GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE, 512).transpose(1,0,2)
			
    pca_trainer_list = []
    pca_all = PCA(n_components=1)
    pca_all.fit(feature_map_PCA.reshape(-1, 512))
    for single_pixel in range(GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE):
        pca = PCA(n_components=1)
        pca.fit(feature_map_PCA[single_pixel])
        pca_trainer_list.append(pca)
    return pca_all, pca_trainer_list   
	
if __name__ == '__main__':
    start=time.time()
    all_pca, pca_list = define_trainer()

    all_feature_map_list = []
    name_list = []
		
    OUT_IMAGE_DIR = './out_2/'
    if not os.path.exists(OUT_IMAGE_DIR):
        os.makedirs(OUT_IMAGE_DIR) 
			
    for name in os.listdir(IMAGE_DIR):
        name_list.append(name)
        all_feature_map_list.append(get_feat_map(FEATURE_MAP_DIR, name))
		
    N_images = len(name_list)
    feature_map_PCA = np.asarray(all_feature_map_list).reshape(N_images, GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE, 512).transpose(1,0,2)
	
    #init PCA
    out_pca_list = []
    for single_pixel in range(GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE):
        out_pca = all_pca.transform(feature_map_PCA[single_pixel])
        out_pca_list.append(out_pca)
    out_pca_list = np.asarray(out_pca_list).reshape(GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE, N_images).transpose(1,0)
			
    end=time.time()
    print('cost time is %.1fseconds.' % (end-start))
    print("start to write files")
    print("length is %d"%len(name_list))
#     for idx in range(len(name_list)):
#         print(out_pca_list[idx])
#         write_one_pca_feature_map(OUT_IMAGE_DIR, name_list[idx], out_pca_list[idx])
    end2=time.time()
    print("time of write files is %.1fseconds." %(end2-end))
    