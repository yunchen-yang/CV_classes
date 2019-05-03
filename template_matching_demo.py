import cv2
from numpy import *
import numpy as np
import math

img_o=np.array([[26, 3, 184, 75, 80, 128, 72, 0, 84],[89, 65, 0, 200, 224, 18, 170, 26, 54],[47, 75, 127, 52, 94, 26, 68, 43, 199],[81, 87, 86, 0, 97, 3, 9, 208, 218],[23, 12, 188, 176, 180, 1, 2, 6, 3],[0, 80, 54, 39, 31, 22, 40, 9, 2],[5, 21, 9, 12, 98, 176, 211, 105, 9]])
img=img_o.astype('uint8')
tem_o=np.array([[3,10,20],[18,1,5],[2,30,3]])
tem=tem_o.astype('uint8')
match_sdd=cv2.matchTemplate(img,tem,cv2.TM_SQDIFF)
match_ncc=cv2.matchTemplate(img,tem,cv2.TM_CCOEFF_NORMED)
sum = 0
for x in range(0,3):
    for y in range(0,3):
        R=(tem_o[x,y]-img_o[x,y])**2
        sum=sum+R
        
i_av=mean(img_o[1:4,1:4])
t_av=mean(tem_o[0:3,0:3])
i_var=np.var(img_o[1:4,1:4])
t_var=np.var(tem_o[0:3,0:3])

ss=0
for i in range(0,3):
    for j in range(0,3):
        rr=((tem_o[i,j]-t_av)*(img_o[i+1,j+1]-i_av))/sqrt(i_var*t_var*81)
        ss=ss+rr
    
        
        
        
        
        