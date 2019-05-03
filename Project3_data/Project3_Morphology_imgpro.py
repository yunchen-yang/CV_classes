import cv2
import numpy as np
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
    
def dilation(img,knl,ori_pos):
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
            if (knl_match+knl>255).any():
                img_pro[x,y]=255
    return img_pro

def erosion(img,knl,ori_pos):
    img_pro=copy.deepcopy(img)
    size_img=img.shape
    size_knl=knl.shape
    x_decenter1=ori_pos[0]
    x_decenter2=size_knl[0]-1-ori_pos[0]
    y_decenter1=ori_pos[1]
    y_decenter2=size_knl[1]-1-ori_pos[1]
    knl_match=np.zeros(knl.shape).astype('uint64')
    xlist=np.array(np.where(img==255))[0,:]
    ylist=np.array(np.where(img==255))[1,:]
    for n in range(0,len(xlist)):
        x=xlist[n]
        y=ylist[n]
        if x<x_decenter1 or y<y_decenter1 or x+x_decenter2>size_img[0]-1 or y+y_decenter2>size_img[1]-1:
            continue
        else:
            for i in range(0,size_knl[0]):
                for j in range(0,size_knl[1]):
                    knl_match[i,j]=img[x-x_decenter1+i,y-y_decenter1+j]
                    knl_match=knl_match*knl*(1/255)
        if (knl_match==knl).all():
            img_pro[x,y]=255
        else:
            img_pro[x,y]=0
    return img_pro

def opening(img,knl,ori_pos):
    img_1=erosion(img,knl,ori_pos)
    img_opening=dilation(img_1,knl,ori_pos)
    return img_opening
    
def closing(img,knl,ori_pos):
    img_1=dilation(img,knl,ori_pos)
    img_closing=erosion(img_1,knl,ori_pos)
    return img_closing
        
path=sys.path[0]+'\\' 
img_no=cv2.imread(path+"original_imgs\\noise.jpg",0)
knl_dim=3
ori=[int((knl_dim-1)/2),int((knl_dim-1)/2)]
knl_round=round_shape(knl_dim)

res_noise1=opening(closing(img_no,knl_round,ori),knl_round,ori)
res_noise2=closing(opening(img_no,knl_round,ori),knl_round,ori)
res_bound1=res_noise1-erosion(res_noise1,knl_round,ori)
res_bound2=res_noise2-erosion(res_noise2,knl_round,ori)

cv2.imwrite(path+"result_imgs\\res_noise1.jpg",res_noise1)
cv2.imwrite(path+"result_imgs\\res_noise2.jpg",res_noise2)
cv2.imwrite(path+"result_imgs\\res_bound1.jpg",res_bound1)
cv2.imwrite(path+"result_imgs\\res_bound2.jpg",res_bound2)