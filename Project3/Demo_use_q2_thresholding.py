import cv2
import numpy as np
import sys

path=sys.path[0]+'/' 
img_mat=cv2.imread(path+"original_imgs/segment.jpg",0)
img_x=img_mat.shape[0]
img_y=img_mat.shape[1]
S1=8
S2=15
T=145
R=0.15
gap_x=int(img_x/S1)
gap_y=int(img_y/S2)
img_t=np.zeros(img_mat.shape).astype('uint8')
for i in range(0,S1):
    for j in range(0,S2):
        if i<S1-1 and j<S2-1:
            ind_x=range(i*gap_x,(i+1)*gap_x)
            ind_y=range(j*gap_y,(j+1)*gap_y)
        elif i==S1-1 and j<S2-1:
            ind_x=range(i*gap_x,img_x)
            ind_y=range(j*gap_y,(j+1)*gap_y)
        elif i<S1-1 and j==S2-1:
            ind_x=range(i*gap_x,(i+1)*gap_x)
            ind_y=range(j*gap_y,img_y)
        elif i==S1-1 and j==S2-1:
            ind_x=range(i*gap_x,img_x)
            ind_y=range(j*gap_y,img_y)
        point=[]
        for x in ind_x:
            for y in ind_y:
                point.append(np.array([x,y,img_mat[x,y]]))
        ave=np.mean(np.array(list(filter((lambda x:x[2]<T),point)))[:,2])
        point_select=list(filter((lambda x:x[2]>=T),point))
        if point_select!=None:
            for n in range(0,len(point_select)):
                if point_select[n][2]>=ave*(1+R):
                    img_t[point_select[n][0],point_select[n][1]]=255
                else:
                    continue
        else:
            continue

cv2.imshow('img_thre_op',img_t)