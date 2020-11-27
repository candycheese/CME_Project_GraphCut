from sklearn.decomposition import PCA
import cv2
from PIL import Image  
import os
import numpy as np
#from skimage import measure,color
import maxflow
import json

GLOBAL_FEATURE_MAP_SIZE = 25
FEATURE_MAP_DIR = './201112_out/test/'
IMAGE_DIR = './201112/'
use_graphcut=False

def get_feat_map(dir, picname):
	feat_map=np.asarray(open(dir+picname+'.txt', 'r').read().split(), dtype=np.float32)
	# print(feat_map)
	return feat_map.reshape(GLOBAL_FEATURE_MAP_SIZE * GLOBAL_FEATURE_MAP_SIZE, 50)

def blend_color(start_color, end_color, f):
	r1, g1, b1 = start_color
	r2, g2, b2 = end_color
	r = r1 + (r2 - r1)*f
	g = g1 + (g2 - g1)*f
	b = b1 + (b2 - b1)*f
	return r,g,b
	
def write_one_pca_feature_map(OUT_IMAGE_DIR, name, feature_map):

	Zconcat_target = Image.new('RGB', (256*2, 256))
	zero=np.zeros((GLOBAL_FEATURE_MAP_SIZE, GLOBAL_FEATURE_MAP_SIZE),dtype = np.uint8)
	_max = 100.
	_min = 30.
	feature_map = feature_map*(feature_map>30.)
	f = (feature_map - _min) / (_max-_min)


	w = open(OUT_IMAGE_DIR + name+'.txt', 'w')
	
	f = np.int32(f>0.0)*f
	for i in f:
		w.write(str(i)+'\n')
	w.close()

	print(np.min(f),np.max(f), np.min(feature_map), np.max(feature_map))

	r = Image.fromarray(f.reshape(GLOBAL_FEATURE_MAP_SIZE,GLOBAL_FEATURE_MAP_SIZE)*255).convert('L')
	g = Image.fromarray(zero).convert('L')
	b = Image.fromarray(zero).convert('L')
	#a = Image.fromarray(n.reshape(GLOBAL_FEATURE_MAP_SIZE,GLOBAL_FEATURE_MAP_SIZE)*70).convert('L')
	high_light=Image.merge("RGB", (r, g, b)) 	
	#origin = Image.open(IMAGE_DIR + name).convert('RGBA')
	#origin.paste(high_light, (0,0), mask=high_light) 
	#origin.save(OUT_IMAGE_DIR + name)
	high_light=high_light.resize((256,256))

	origin = Image.open(IMAGE_DIR + name).convert('RGB')
	Zconcat_target.paste(high_light, (256,0))
	Zconcat_target.paste(origin, (0,0))
	Zconcat_target.save(OUT_IMAGE_DIR + name)

def define_trainer():
	#labels = open('test_result.txt','r')
	#labels_table = json.load(labels)
	#training PCA
	all_feature_map_list = []
	name_list = []
	pca_feature_dir = FEATURE_MAP_DIR
	#pca_feature_list = os.listdir(pca_feature_dir)
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
	feature_map_PCA = np.asarray(all_feature_map_list).reshape(N_images, GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE, 50).transpose(1,0,2)
			
	pca_trainer_list = []
	pca_all = PCA(n_components=1)
	pca_all.fit(feature_map_PCA.reshape(-1, 50))
	for single_pixel in range(GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE):
		pca = PCA(n_components=1)
		pca.fit(feature_map_PCA[single_pixel])
		pca_trainer_list.append(pca)
	return pca_all, pca_trainer_list
	
if __name__ == '__main__':
	all_pca, pca_list = define_trainer()

	all_feature_map_list = []
	name_list = []
		
	OUT_IMAGE_DIR = './out_1/'
	if not os.path.exists(OUT_IMAGE_DIR):
		os.makedirs(OUT_IMAGE_DIR) 
			
	for name in os.listdir(IMAGE_DIR):
		# print(name)
		name_list.append(name)
		all_feature_map_list.append(get_feat_map(FEATURE_MAP_DIR, name))

	print(len(name_list))
	N_images = len(name_list)
	feature_map_PCA = np.asarray(all_feature_map_list).reshape(N_images, GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE, 50).transpose(1,0,2)
	
	#init PCA
	out_pca_list = []
	for single_pixel in range(GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE):
		out_pca = all_pca.transform(feature_map_PCA[single_pixel])
		out_pca_list.append(out_pca)
	out_pca_list = np.asarray(out_pca_list).reshape(GLOBAL_FEATURE_MAP_SIZE*GLOBAL_FEATURE_MAP_SIZE, N_images).transpose(1,0)
			
	for idx in range(len(name_list)):
		write_one_pca_feature_map(OUT_IMAGE_DIR, name_list[idx], out_pca_list[idx])

