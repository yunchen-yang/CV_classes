import cv2
import numpy as np
import copy
import sys

def point_detecting(img_input,filter_kernel):
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
                            for x in range(-decenter,decenter+1):
                                for y in range(-decenter,decenter+1):
                                    G[decenter+x,decenter+y]=H[decenter+x,decenter+y]*filter_kernel[decenter+x,decenter+y]
                            new_img[i,j]=sum(sum(G))
                        else:
                            new_img[i,j]=0
    return new_img

def thresholding(img_mat,T):
    img_t=np.zeros(img_mat.shape).astype('uint8')
    for x in range(0,img_mat.shape[0]):
        for y in range(0,img_mat.shape[1]):
            if abs(img_mat[x,y])>=T:
                img_t[x,y]=255
            else:
                continue
    return img_t
    
path=sys.path[0]+'\\' 
img=cv2.imread(path+"original_imgs\\point.jpg",0)
img_color=cv2.imread(path+"original_imgs\\point.jpg")
point_knl=np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,24,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]])
img_mask=point_detecting(img,point_knl)
ratio=(np.max(img_mask)-np.min(img_mask))/255

img_thre=thresholding(img_mask,700)
img_draw=copy.deepcopy(img_color)
point_coor=np.where(img_thre==255)
for num in range(len(point_coor[0])):
    if point_coor[0][num]>40 and point_coor[0][num]<430 and point_coor[1][num]>5 and point_coor[1][num]<352:
        img_draw=cv2.circle(img_draw,(point_coor[1][num],point_coor[0][num]),5,(0,0,255),1)
        print([point_coor[0][num],point_coor[1][num]])  
cv2.imwrite(path+"result_imgs\\detected_points.jpg",img_draw)

