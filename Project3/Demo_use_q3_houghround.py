import cv2
import numpy as np
import sys
import math
import copy

path=sys.path[0]+'\\' 
img=cv2.imread(path+"original_imgs\\hough.jpg",0)
img_color=cv2.imread(path+"original_imgs\\hough.jpg")
bimap=cv2.Canny(img,100,200)
R1=15
R2=35

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