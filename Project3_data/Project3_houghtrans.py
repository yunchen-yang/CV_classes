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
                
def hough_draw(img,bimap,votemap,T,t_rho,t_theta,step):
    votemap_noedge=votemap[1:votemap.shape[0]-1,1:votemap.shape[1]-1]
    pl_coor=np.where(votemap_noedge>T)
    img_draw=copy.deepcopy(img)
    A=int(math.sqrt(pow(img.shape[0],2)+pow(img.shape[1],2)))
    parameter=[]
    for cnt in range(len(pl_coor[0])):
        x=pl_coor[0][cnt]+1
        y=pl_coor[1][cnt]+1
        localmat=votemap[x-1:x+2,y-1:y+2]
        if votemap[x,y]==np.max(localmat):
            rho=x+0.5-A
            theta=(math.pi/2)*(3*y/step-1)
            parameter.append([rho,theta,votemap[x,y]])
        else:
            continue
    selected=[]
    for num in range(0,len(parameter)):
        if len(selected)==0:
            selected.append(parameter[num])
        else:
            for cnt in range(0,len(selected)):
                if parameter[num][0]>selected[cnt][0]-t_rho and parameter[num][0]<selected[cnt][0]+t_rho \
                and parameter[num][1]>selected[cnt][1]-t_theta and parameter[num][1]<selected[cnt][1]+t_theta:
                    if parameter[num][2]>selected[cnt][2]:
                        selected[cnt]=parameter[num]
                    break
            else:
                selected.append(parameter[num])
    for dn in range(0,len(selected)):
        rho=selected[dn][0]
        theta=selected[dn][1]            
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
        img_draw=cv2.line(img_draw,(Y1-1,X1-1),(Y2-1,X2-1),(0,255,0),2)        
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

votemap_neg45=hough_votemap(bimap_neg45,540)
img_neg45=hough_draw(img_color,bimap_neg45,votemap_neg45,30,15,math.pi/30,540)
cv2.imwrite(path+"result_imgs\\blue_lines.jpg",img_neg45)

votemap_vertical=hough_votemap(bimap_vertical,300)
img_vertical=hough_draw(img_color,bimap_vertical,votemap_vertical,180,15,math.pi/30,300)
cv2.imwrite(path+"result_imgs\\red_lines.jpg",img_vertical)

cv2.imwrite(path+"result_imgs\\bimap_blue_lines.jpg",bimap_neg45)
cv2.imwrite(path+"result_imgs\\bimap_red_lines.jpg",bimap_vertical)

