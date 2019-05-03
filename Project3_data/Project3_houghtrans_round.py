import cv2
import numpy as np
import sys
import math
import copy

def round_hough(bimap,R1,R2):
    coor=np.where(bimap==255)
    x=bimap.shape[0]
    y=bimap.shape[1]
    vote=np.zeros([x,y,(R2-R1)]).astype('int')
    for n in range(0,x):
        for m in range(0,y):
            for cnt in range(0,len(coor[0])):
                x1=coor[0][cnt]
                y1=coor[1][cnt]    
                r=math.sqrt(pow((n-x1),2)+pow((m-y1),2))
                if round(r)<R2 and round(r)>=R1:
                    vote[n,m,(round(r)-R1)]=vote[n,m,(round(r)-R1)]+1
                else:
                    continue
    return vote

def hough_draw_round(img,votemap,R1,R2,T,t_a,t_b,t_r):
    votemap_noedge=votemap[1:votemap.shape[0]-1,1:votemap.shape[1]-1,1:votemap.shape[2]-1]
    pl_coor=np.where(votemap_noedge>T)
    img_draw=copy.deepcopy(img)
    parameter=[]
    for cnt in range(0,len(pl_coor[2])):
        x=pl_coor[0][cnt]+1
        y=pl_coor[1][cnt]+1
        z=pl_coor[2][cnt]+1
        localmat=votemap[x-1:x+2,y-1:y+2,z-1:z+2]
        if votemap[x,y,z]==np.max(localmat):
            parameter.append([x,y,z+R1])
        else:
            continue            
    selected=[]
    for num in range(0,len(parameter)):
        if len(selected)==0:
            selected.append(parameter[num])
        else:
            for cnt in range(0,len(selected)):
                if parameter[num][0]>selected[cnt][0]-t_a and parameter[num][0]<selected[cnt][0]+t_a \
                and parameter[num][1]>selected[cnt][1]-t_b and parameter[num][1]<selected[cnt][1]+t_b\
                and parameter[num][2]>selected[cnt][1]-t_r and parameter[num][1]<selected[cnt][1]+t_r:
                    if parameter[num][2]>selected[cnt][2]:
                        selected[cnt]=parameter[num]
                    break
            else:
                selected.append(parameter[num])                
    for dn in range(0,len(selected)):               
        img_draw=cv2.circle(img_draw,(selected[dn][1],selected[dn][0]),selected[dn][2],(0,0,255),2)
    return img_draw

path=sys.path[0]+'\\' 
img=cv2.imread(path+"original_imgs\\hough.jpg",0)
img_color=cv2.imread(path+"original_imgs\\hough.jpg")
img_blur=cv2.GaussianBlur(img,(7,7),1)
map_edge=cv2.Canny(img_blur,100,250)

votemap_edge=round_hough(map_edge,20,30)
img_d=hough_draw_round(img_color,votemap_edge,20,30,50,5,5,5)
cv2.imwrite(path+"result_imgs\\coin.jpg",img_d)