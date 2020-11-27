
#from sklearn.decomposition import PCA  
import cv2
from PIL import Image  
import os
import numpy as np
#from skimage import measure,color
import maxflow
fe_map_size=25
use_grab=False
use_graphcut=True

def img_polar_transform(input_img,center,r_range,theta_rangge=(0,360),r_step=0.5,theta_step=360.0/(90.)):
    minr,maxr=r_range
    mintheta,maxtheta=theta_rangge
    H=int((maxr-minr)/r_step+1)
    W=int((maxtheta-mintheta)/theta_step+1)
    #print(H,W,theta_step)
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
            if ((px>=0 and px<=255) and (py>=0 and py<=255)):
            	#print(px,py)
                output_img[i,j]=input_img[px,py]
    return output_img
def create_graph_and_cut(image, mask):
	#print(origin_image)
	g = maxflow.Graph[float]()
	nodes = g.add_grid_nodes(image.shape[:2])

	bottom_weights = np.zeros(image.shape[:2])
	left_weighta = np.zeros(image.shape[:2])
   
	bottom_weights = 100. / np.exp(((image[:-1,:] - image[1:,:]) ** 2) / (2 * 1 ** 2))
	left_weights = 100. /np.exp(((image[:,:-1] - image[:,1:]) ** 2) / (2 * 1 ** 2))
	#np.max(bottom_weights)
	#print bottom_weights
#	  print(((image[:-1,:,:] - image[1:,:,:]) ** 2).sum(axis=2) / (2 * sigma ** 2))
#	  print(((image[:-1,:,:] - image[1:,:,:]) ** 2).sum(axis=2) / (2 * sigma ** 2))
   
	# bottom_weights = ((image[:-1,:,:] - image[1:,:,:]) ** 2).sum(axis=2)
	# left_weights = ((image[:,:-1,:] - image[:,1:,:]) ** 2).sum(axis=2)
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

	g.add_grid_tedges(nodes, (1-mask)*5., (mask)+image*0.1)
	#plot_graph_2d(g, nodes.shape, plot_terminals=False)
	#assert(0)
	curr_maxflow = g.maxflow()
	return g.get_grid_segments(nodes)



def create_graph_and_cut_refine(image, mask):

	g = maxflow.Graph[float]()
	nodes = g.add_grid_nodes(image.shape[:2])

	bottom_weights = np.zeros(image.shape[:2])
	left_weighta = np.zeros(image.shape[:2])
   
	bottom_weights = 2. / np.exp(((image[:-1,:] - image[1:,:]) ** 2) / (2 * 10. ** 2))
	left_weights = 2. /np.exp(((image[:,:-1] - image[:,1:]) ** 2) / (2 * 10. ** 2))
	#np.max(bottom_weights)
	#print bottom_weights
#	  print(((image[:-1,:,:] - image[1:,:,:]) ** 2).sum(axis=2) / (2 * sigma ** 2))
#	  print(((image[:-1,:,:] - image[1:,:,:]) ** 2).sum(axis=2) / (2 * sigma ** 2))
   
	# bottom_weights = ((image[:-1,:,:] - image[1:,:,:]) ** 2).sum(axis=2)
	# left_weights = ((image[:,:-1,:] - image[:,1:,:]) ** 2).sum(axis=2)
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
	#mask = np.int32(mask)*100
	g.add_grid_tedges(nodes, (1-mask)*1.8, mask*0.7+image*0.08)
	#plot_graph_2d(g, nodes.shape, plot_terminals=False)
	#assert(0)
	curr_maxflow = g.maxflow()
	return g.get_grid_segments(nodes)


def compute_max_length(img):
	num_theta = img.shape[1]

	out_all = np.ones(img.shape[1])*(img.shape[0]-1)
	#out_all = np.ones((img.shape[0]))
	for i in range(num_theta):
		index = np.where(img[:,i]>10)
		
		#1/0
		try:
			#print(np.max(index))
			out_all[i] = np.max(index)
		except:
			pass
			#out_all[i] = 0.
		#if out_all[i]<10:
			
	return out_all

def denoise_p(mask_image, mask_theta, center,r_range,theta_rangge=(0,360),r_step=0.5,theta_step=360.0/(180*32)):
    minr,maxr=r_range
    mintheta,maxtheta=theta_rangge
    H=int((maxr-minr)/r_step+1)
    W=int((maxtheta-mintheta)/theta_step+1)
    #print(H,W,theta_step)
    #output_img=np.zeros((256,256),input_img.dtype)
    x_center,y_center=center
    
    #r,theta=cv2.polarToCart(x,y,angleInDegrees=True)
    mask_image_out = np.zeros((256,256))
    for x in range(256):
        for y in range(256):
        	px = x - x_center
        	py = y - y_center
        	r,theta=cv2.cartToPolar(px,py,angleInDegrees=True)
        	#print(px, py, r[0][0],theta[0][0])
        	if int(theta[0,0]/theta_step)<=mask_theta[1] and int(theta[0,0]/theta_step)>=mask_theta[0]:
        		mask_image_out[x,y] = mask_image[x,y]*255
        	else:
        		continue
        		mask_image_out[x,y] = 0
    return mask_image_out			
			

def denoise(mask_image, max_mask, center,r_range,theta_rangge=(0,360),r_step=0.5,theta_step=360.0/(90.)):
    minr,maxr=r_range
    mintheta,maxtheta=theta_rangge
    H=int((maxr-minr)/r_step+1)
    W=int((maxtheta-mintheta)/theta_step+1)
    #print(H,W,theta_step)
    #output_img=np.zeros((256,256),input_img.dtype)
    x_center,y_center=center
    
    for x in range(256):
        for y in range(256):
            px = x - x_center
            py = y - y_center
            r,theta=cv2.cartToPolar(px,py,angleInDegrees=True)
            now_r = max_mask[int(theta[0,0]/theta_step)]*r_step
            if r[0,0] > now_r or now_r < 30:
                mask_image[x,y] = 0.


    return mask_image

def blend_color(start_color, end_color, f):
	r1, g1, b1 = start_color
	r2, g2, b2 = end_color
	r = r1 + (r2 - r1)*f
	g = g1 + (g2 - g1)*f
	b = b1 + (b2 - b1)*f
	return r,g,b
	
def get_files(now_dir):
	out=[]
	for i in os.listdir(now_dir):
		if not os.path.isdir(i):
			out.append(i[:-4])
	return out
	
def get_mask_data(x_center,y_center,r_max):
	#x_center,y_center,r_max = parms
	mask_image = np.zeros((256, 256))
	minr,maxr=0,256
	for x in range(256):
		for y in range(256):
			px = x - x_center/1024*256
			py = y - y_center/1024*256
			r,theta=cv2.cartToPolar(px,py,angleInDegrees=True)
			if r[0,0]<=r_max/1024*256:
				mask_image[x,y] = 0
			else:
				mask_image[x,y] = 255
				
	return mask_image
'''
names=get_files('./local/4_test/')

out_map=[]
for i in range(len(names)):
	feat_map=np.asarray(open('./local/4_test/'+names[i]+'.txt','r').read().split(),dtype=np.float32)
	out_map.append(feat_map.reshape(fe_map_size*fe_map_size,50))
		#print("test acc = {:.4f}".format(step, acc))
out_map=np.asarray(out_map).reshape(len(names),fe_map_size*fe_map_size,50).transpose(1,0,2)
pca = PCA(n_components=1)
out=[]
for i in range(fe_map_size*fe_map_size):
	out_f=pca.fit_transform(out_map[i])
	out.append(out_f)
out=np.asarray(out).reshape(fe_map_size*fe_map_size,len(names)).transpose(1,0)
		#out=int(out>0)
'''
def get_param(name):
	if '2012-03-07' in name:
		return 518.20599, 532.49500, 52.328029
	if '2012-06-14' in name:
		return 504.76401, 490.51700, 52.043289
	return None



OUT_IMAGE_DIR = './out_3/'
if not os.path.exists(OUT_IMAGE_DIR):
	os.makedirs(OUT_IMAGE_DIR) 
orgin_image = './orgin'
input_polar_images_dir = './out_1/'
file_list = os.listdir(input_polar_images_dir)
#not_name = ['18_2', '18_3', '18_4','18_5','19_1','19_2','19_3','19_4']
#mask_array_3 = get_mask_data(518.20599, 532.49500, 52.328029)
#mask_array_6 = get_mask_data(504.76401, 490.51700, 52.043289)
for file_idx in range(len(file_list)):

    polar_image_name = file_list[file_idx]
    print(polar_image_name)
    if 'txt' in polar_image_name or 'flow' in polar_image_name:
        continue
    concat_image = cv2.imread(input_polar_images_dir + polar_image_name)
    concat_image= cv2.split(concat_image)[2] 
	
    origin_image_save = concat_image[:,0:256]
    #origin_image_save = cv2.medianBlur(origin_image_save, 5)
    origin_image = np.uint8(np.clip(origin_image_save - np.mean(origin_image_save), 0., 255.))

    mask = concat_image[:,256:]
    h,w = origin_image.shape[:2]
    center = (int(h/2),int(w/2))
    r_range = (0,200)
    mask_polar = img_polar_transform(mask, center,r_range)
    origin_max_length = compute_max_length(mask_polar)

    if use_graphcut:
        mask = np.float32(mask)/255.

        sgm = create_graph_and_cut(origin_image, mask)

        mask2 = np.uint8(np.logical_not(sgm)==0)
        mask2 = denoise(mask2, origin_max_length, center=center, r_range=r_range)
        sgm_refine = create_graph_and_cut_refine(origin_image, mask2)
        mask3 = np.uint8(np.logical_not(sgm_refine)==0)
		
        Zconcat_target = Image.new('RGB', (256*2, 256))
        r  =Image.fromarray(mask3*255).convert('L')
        g = Image.fromarray(np.zeros((256,256))).convert('L')
        b = Image.fromarray(np.zeros((256,256))).convert('L')
        high_light=Image.merge("RGB", (r, g, b))
        origin_image = Image.fromarray(origin_image_save).convert('L')
        origin_image = origin_image.convert('RGB')	
        
        Zconcat_target.paste(high_light, (256,0))
        Zconcat_target.paste(origin_image, (0,0))
        Zconcat_target.save('./out_3/'+polar_image_name)

