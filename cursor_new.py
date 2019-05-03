import cv2
from numpy import *
import numpy as np
import math
##
##define sampling function
def sampling(image_original,ratio):
    size_original=image_original.shape
    length=size_original[0]
    width=size_original[1]
    length_adj=math.floor(length/ratio)
    width_adj=math.floor(width/ratio)
    rgb_or_gray=len(size_original)
    if rgb_or_gray==3:
        sampled_img=zeros((length_adj,width_adj,size_original[2]))
        for colorchannel in range(0,size_original[2]):
            for x_adj in range(0,length_adj):
                for y_adj in range(0,width_adj):
                    sampled_img[x_adj,y_adj,colorchannel]=np.mean(image_original[math.floor(ratio*x_adj):math.floor(ratio*(x_adj+1)),math.floor(ratio*y_adj):math.floor(ratio*(y_adj+1)),colorchannel])
    else:
        sampled_img=zeros((length_adj,width_adj))
        for x_adj in range(0,length_adj):
                for y_adj in range(0,width_adj):
                    sampled_img[x_adj,y_adj]=np.mean(image_original[math.floor(ratio*x_adj):math.floor(ratio*(x_adj+1)),math.floor(ratio*y_adj):math.floor(ratio*(y_adj+1))])

    sampled_img_output=np.rint(sampled_img).astype('uint8')
    return sampled_img_output

img_n_o=cv2.imread("/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/neg_4.jpg")
img_p_o=cv2.imread("/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/pos_6.jpg")
img_p=cv2.imread("/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/pos_6.jpg")
#img_p=cv2.cvtColor(img_p_o,cv2.COLOR_BGR2GRAY)
#img_p=cv2.GaussianBlur(img_p,(3,3),sqrt(2))

temo=cv2.imread("/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/template.png")
tem=temo[6:30,5:19]
#tem2=cv2.imread("/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/template2.png",0)
#tem3=cv2.imread("/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/template3.jpg",0)

length_tem=tem.shape[0]
width_tem=tem.shape[1]

length_tema=math.floor(length_tem/2)
width_tema=math.floor(width_tem/2)
tem_re=sampling(tem,2)

match=cv2.matchTemplate(img_p,tem_re,cv2.TM_CCOEFF_NORMED)
#match2=cv2.matchTemplate(img_p,tem2,cv2.TM_SQDIFF_NORMED)
#match3=cv2.matchTemplate(img_p,tem3,cv2.TM_SQDIFF_NORMED)
#match=match2+match3

score_max=[cv2.minMaxLoc(match)[1]]
loc_max=[cv2.minMaxLoc(match)[3]]

cv2.rectangle(img_p_o,(loc_max[0][0],loc_max[0][1]),(loc_max[0][0]+width_tema,loc_max[0][1]+length_tema),(0,0,255),2)

cv2.imshow("cursor",img_p_o)
cv2.waitKey(0)
cv2.destroyAllWindows()