import cv2
from numpy import *
import numpy as np
import math
import copy
import sys

def round_shape(dim):
    if dim%2==0:
        print("dimension of matrix must be odd")
        return None
    mat=np.zeros([dim,dim])
    radius=dim/2
    for x in range(0,dim):
        for y in range(0,dim):
            if pow(((x-(dim-1)/2)*(x-(dim-1)/2)+(y-(dim-1)/2)*(y-(dim-1)/2)),0.5)<radius:
                mat[x,y]=255
    mat=mat.astype('uint64')            
    return mat
    
path=sys.path[0]+'/' 
img_ori=cv2.imread(path+"original_imgs/noise.jpg",0)
img=cv2.imread(path+"original_imgs/noise.jpg",0)
knl_dim=7
ori_pos=[int((knl_dim-1)/2),int((knl_dim-1)/2)]
knl=round_shape(knl_dim)

img_pro=copy.deepcopy(img)
size_img=img.shape
size_knl=knl.shape
x_decenter1=ori_pos[0]
x_decenter2=size_knl[0]-1-ori_pos[0]
y_decenter1=ori_pos[1]
y_decenter2=size_knl[1]-1-ori_pos[1]
knl_match=np.zeros(knl.shape).astype('uint64')
for x in range(x_decenter1,size_img[0]-x_decenter2):
    for y in range(y_decenter1,size_img[1]-y_decenter2):
        for i in range(0,size_knl[0]):
            for j in range(0,size_knl[1]):
                knl_match[i,j]=img[x-x_decenter1+i,y-y_decenter1+j]
                
        if ((knl_match+knl)>255).any():
            img_pro[x,y]=255

cv2.imshow('img_pro',img_pro)
cv2.imshow('img',img)
test=(img==img_pro).all()