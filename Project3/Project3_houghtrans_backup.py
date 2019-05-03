import cv2
import numpy as np
import sys
import math
import copy

def line_detecting(img_input,filter_kernel):
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

def thresholding(img_mat,T,direction):
    img_t=np.zeros(img_mat.shape).astype('uint8')
    if direction==1:
        for x in range(0,img_mat.shape[0]):
            for y in range(0,img_mat.shape[1]):
                if abs(img_mat[x,y])>=T:
                    img_t[x,y]=255
                else:
                    continue
    elif direction==-1:
        for x in range(0,img_mat.shape[0]):
            for y in range(0,img_mat.shape[1]):
                if abs(img_mat[x,y])<=T:
                    img_t[x,y]=255
                else:
                    continue
    else:
        for x in range(0,img_mat.shape[0]):
            for y in range(0,img_mat.shape[1]):
                if abs(img_mat[x,y])<=T[1] and abs(img_mat[x,y])>=T[0]:
                    img_t[x,y]=255
                else:
                    continue
    return img_t
 
def score2map(img_score):
    ratio=(np.max(img_score)-np.min(img_score))/255
    img_map=((img_score-np.min(img_score))/ratio).astype('uint8')
    return img_map

def hough_func(theta,x,y):
    X=x+1
    Y=y+1
    rho=math.cos(theta)*X+math.sin(theta)*Y
    return rho

def hough_votemap(bimap,step):
    coor=np.where(bimap==255)
    coor_x=coor[0]
    coor_y=coor[1]
    num=len(coor_x)
    A=int(math.sqrt(bimap.shape[0]*bimap.shape[0]+bimap.shape[1]*bimap.shape[1]))
    vote=np.zeros([A*2,step]).astype('int')
    for n in range(1,step):
        for m in range(0,num):
            r=hough_func((math.pi/2)*(3*n/step-1),coor_x[m],coor_y[m])
            vote[int(r+A),n]=vote[int(r+A),n]+1
    return vote
                
def hough_draw(img,bimap,votemap,T,T_std,step):
    pl_coor=np.where(votemap>T)
    img_draw=copy.deepcopy(img)

    A=int(math.sqrt(pow(img.shape[0],2)+pow(img.shape[1],2)))
    for cnt in range(len(pl_coor[0])):
        rho=pl_coor[0][cnt]+0.5-A
        theta=(math.pi/2)*(3*pl_coor[1][cnt]/step-1)
        if math.sin(theta)==0:
            X1=int(rho/math.cos(theta))
            X2=X1
            Y1=1
            Y2=img.shape[1]+1
        else:
            X1=1
            X2=img.shape[0]+1
            Y1=int((rho-math.cos(theta)*X1)/math.sin(theta))
            Y2=int((rho-math.cos(theta)*X2)/math.sin(theta))
        map_ol1=(bimap/255).astype('uint8')
        map_ol2=np.zeros(bimap.shape).astype('uint8')
        map_ol2=cv2.line(map_ol2,(Y1-1,X1-1),(Y2-1,X2-1),1,1)
        map_ol=map_ol1+map_ol2
        coor_select=np.where(map_ol==2)
        if len(coor_select[0])>T:
            dotgroup=np.zeros([len(coor_select[0]),1])
            for cnt2 in range(0,len(coor_select[0])):
                dotgroup[cnt2,0]=math.sqrt(pow((coor_select[0][cnt2]-(rho/math.cos(theta))),2)+pow(coor_select[1][cnt2],2))
            dot_std=np.std(dotgroup,axis=0)
            if dot_std<T_std:
                img_draw=cv2.line(img_draw,(Y1-1,X1-1),(Y2-1,X2-1),(0,0,255),1)
        else:
            continue        
    return img_draw             
                    
                            
path=sys.path[0]+'\\' 
img=cv2.imread(path+"original_imgs\\hough.jpg",0)
img_color=cv2.imread(path+"original_imgs\\hough.jpg")
knl_neg45=np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])
knl_vertical=np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
score_neg45=line_detecting(img,knl_neg45)
score_vertical=line_detecting(img,knl_vertical)
map_neg45=score2map(score_neg45)
map_vertical=score2map(score_vertical)
bimap_neg45=thresholding(map_neg45,[70,80],0)
bimap_vertical=thresholding(map_vertical,[50,70],0)

votemap_neg45=hough_votemap(bimap_neg45,2700)
img_neg45=hough_draw(img_color,bimap_neg45,votemap_neg45,30,30,2700)
cv2.imshow('img_draw_neg45',img_neg45)

votemap_vertical=hough_votemap(bimap_vertical,300)
img_vertical=hough_draw(img_color,bimap_vertical,votemap_vertical,180,95,300)
cv2.imshow('img_draw_vertical',img_vertical)

cv2.imshow('img',img)
cv2.imshow('map_neg45',map_neg45)
cv2.imshow('map_vertical',map_vertical)
cv2.imshow('bimap_neg45',bimap_neg45)
cv2.imshow('bimap_vertical',bimap_vertical)

#map_edge=cv2.Canny(img,100,200)
#cv2.imshow('map_edge',map_edge)
#votemap_edge=hough_votemap(map_edge,270)
#img_edge=hough_draw(img_color,map_edge,votemap_edge,150,100,270)
#cv2.imshow('img_edge',img_edge)
