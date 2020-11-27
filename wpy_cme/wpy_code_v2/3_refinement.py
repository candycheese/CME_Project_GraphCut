from sklearn.decomposition import PCA  
import cv2
from PIL import Image  
import os
import numpy as np
# from skimage import measure,color
import maxflow
# import util
fe_map_size=25
use_grab=False
use_graphcut=True

origin_image_dir = '/disk/dataset/cme_matting/test/test_ori/'
origin_image_save_dir = '/disk/dataset/Earlier_Project/refine_ori/'
IMAGE_SIZE = 512
#两次graphcut的效果可能会好点
def create_graph_and_cut(image, mask):

    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes(image.shape[:2])

    bottom_weights = np.zeros(image.shape[:2])
    left_weighta = np.zeros(image.shape[:2])
   
    bottom_weights = 3000. / np.exp(((image[:-1,:] - image[1:,:]) ** 2) / (2 * 1 ** 2))
    left_weights = 3000. /np.exp(((image[:,:-1] - image[:,1:]) ** 2) / (2 * 1 ** 2))

    bottom_weights = np.vstack((bottom_weights, np.zeros(image.shape[1])))
    left_weights = np.hstack((left_weights, np.zeros((image.shape[0], 1))))
   
    bottom_structure = np.array([[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 1, 0]])
    left_structure = np.array([[0, 0, 0],
                               [1, 0, 0],
                               [0, 0, 0]])

    g.add_grid_edges(nodes, bottom_weights, structure=bottom_structure, symmetric=True)
    g.add_grid_edges(nodes, left_weights, structure=left_structure, symmetric=True)
    mask = mask*0.8+0.1
    g.add_grid_tedges(nodes, (1-mask)*1.8, (mask)*0.7+image*0.075)
#     g.add_grid_tedges(nodes, (1-mask)*1.1, (mask)*0.01)
    curr_maxflow = g.maxflow()
    return g.get_grid_segments(nodes)



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

    g.add_grid_tedges(nodes, (1-mask)*1.8, mask*0.7+image*0.075)

    curr_maxflow = g.maxflow()
    return g.get_grid_segments(nodes)


    
def get_files(now_dir):
    out=[]
    for i in os.listdir(now_dir):
        if not os.path.isdir(i):
            out.append(i[:-4])
    return out


OUT_IMAGE_DIR = './out_3/'

if not os.path.exists(OUT_IMAGE_DIR):
    os.makedirs(OUT_IMAGE_DIR) 

input_polar_images_dir = './co_out_2/'
file_list = os.listdir(input_polar_images_dir)

for file_idx in range(len(file_list)):

    polar_image_name = file_list[file_idx]
    # print(polar_image_name)
    # break
    if 'npy' in polar_image_name or 'flow' in polar_image_name:
        continue
    # print(os.getcwd())
    img_path = os.getcwd() + input_polar_images_dir.split('.')[1] + polar_image_name
    # print(img_path)
    # break
    concat_image = cv2.imread(img_path)
    # concat_image = cv2.imread(input_polar_images_dir + polar_image_name)
    print(concat_image.shape)
    concat_image= cv2.split(concat_image)[2] # 拆分通道，提取r通道
    
    origin_image_save = cv2.imread(origin_image_dir+polar_image_name)#concat_image[:,0:256]
    origin_image_save= cv2.split(origin_image_save)[2]
    #origin_image_save = cv2.medianBlur(origin_image_save, 5)
    origin_image = np.uint8(np.clip(origin_image_save - np.mean(origin_image_save)*1.2, 0., 255.))

    mask = cv2.resize(concat_image[:,IMAGE_SIZE:], (IMAGE_SIZE,IMAGE_SIZE))


    if use_graphcut:
    #n=np.float32(out[i])
        mask = np.float32(mask)/255.

        sgm = create_graph_and_cut(origin_image, mask)

        mask2 = np.uint8(np.logical_not(sgm)==0)

        #kernel = np.ones((3,3),np.uint8)  
        #mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        #mask2 = cv2.morphologyExpip install PyMaxflow(mask2, cv2.MORPH_OPEN, kernel)
        #mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        #mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        #mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

        #kernel = np.ones((5,5),np.uint8) 
        #mask2 = cv2.dilate(mask2,kernel,iterations = 2)

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
    save_dir = './out_3_3/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Zconcat_target.save(save_dir+polar_image_name)
