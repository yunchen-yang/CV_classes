import cv2
import numpy as np
import sys
import copy

def thresholding_op(img_mat,T,R,S1,S2):
    img_x=img_mat.shape[0]
    img_y=img_mat.shape[1]
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
    return img_t

def thresholding(img_mat,T):
    img_t=np.zeros(img_mat.shape).astype('uint8')
    for x in range(0,img_mat.shape[0]):
        for y in range(0,img_mat.shape[1]):
            if img_mat[x,y]>=T:
                img_t[x,y]=255
            else:
                continue
    return img_t
    
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
             
path=sys.path[0]+'/' 
img=cv2.imread(path+"original_imgs/segment.jpg",0)
img_color=cv2.imread(path+"original_imgs/segment.jpg")
img_thre145=thresholding(img,145)
img_thre180=thresholding(img,180)
img_thre195=thresholding(img,195)
img_thre210=thresholding(img,210)
img_thre_op=thresholding_op(img,195,0.13,8,15)

knl=round_shape(3)
img_threopop=closing(opening(img_thre_op,knl,[1,1]),knl,[1,1])


x_ind=np.where(img_threopop==255)[0]
y_ind=np.where(img_threopop==255)[1]
x_min=min(x_ind)-10
y_min=min(y_ind)-10
x_max=max(x_ind)+10
y_max=max(y_ind)+10
img_seg=cv2.rectangle(img_color,(y_min,x_min),(y_max,x_max),(0,0,255),2)

cv2.imwrite(path+"result_imgs\\img_thre145.jpg",img_thre145)
cv2.imwrite(path+"result_imgs\\img_thre180.jpg",img_thre180)
cv2.imwrite(path+"result_imgs\\img_thre195.jpg",img_thre195)
cv2.imwrite(path+"result_imgs\\img_thre210.jpg",img_thre210)

cv2.imwrite(path+"result_imgs\\img_adaptive_thresholding.jpg",img_thre_op)
print([x_min,y_min])
print([x_min,y_max])
print([x_max,y_min])
print([x_max,y_max])
cv2.imwrite(path+"result_imgs\\img_segmentation.jpg",img_seg)