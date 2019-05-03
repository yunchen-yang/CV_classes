import cv2
import numpy as np
import copy
import sys

path=sys.path[0]+'/' 
img_input=cv2.imread(path+"original_imgs/point.jpg",0)
filter_kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
size_img_input=img_input.shape
kernel_size=filter_kernel.shape
decenter=int((kernel_size[1]-1)/2)
new_img=np.zeros(img_input.shape)
for i in range(0,size_img_input[0]):
        for j in range(0,size_img_input[1]):
            H=np.zeros((kernel_size[0],kernel_size[1])).astype('int')
            G=np.zeros((kernel_size[0],kernel_size[1])).astype('int')
            for n in range(-decenter,decenter+1):
                for m in range(-decenter,decenter+1):
                    if i+n in range(0,size_img_input[0]) and j+m in range(0,size_img_input[1]):
                        H[decenter+n,decenter+m]=img_input[i+n,j+m]
                    else:
                        H[decenter+n,decenter+m]=0
            for x in range(-decenter,decenter+1):
                for y in range(-decenter,decenter+1):
                    G[decenter+x,decenter+y]=H[decenter+x,decenter+y]*filter_kernel[decenter+x,decenter+y]
            new_img[i,j]=sum(sum(G))
    
#img_new_adjusted=new_img.clip(0, 255)
#img_new=np.rint(img_new_adjusted).astype('uint8')


#cv2.imshow('img_mask',img_new)
