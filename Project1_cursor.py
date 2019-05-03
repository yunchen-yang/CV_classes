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
for i_c in range(1,16):
    dir_str="/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/images/pos/pos_"+str(i_c)+".jpg"
    img_p_o=cv2.imread(dir_str)
    img_p=cv2.imread(dir_str)

    temo=cv2.imread("/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/template.png")
    tem=temo[6:30,5:19]

    length_tem=tem.shape[0]
    width_tem=tem.shape[1]

    size_list=[]
    n_size=0
    score_list=[]
    loc_list=[]
    for ind in range(10,22):
        length_tema=math.floor(length_tem/ind*10)
        width_tema=math.floor(width_tem/ind*10)
        size_list.append([length_tema,width_tema])
        tem_re=sampling(tem,ind/10)
        match=cv2.matchTemplate(img_p,tem_re,cv2.TM_CCOEFF_NORMED)
        score_m=[cv2.minMaxLoc(match)[1]]
        loc_m=[cv2.minMaxLoc(match)[3]]
        score_list.append(score_m)
        loc_list.append(loc_m)

    max_ind=cv2.minMaxLoc(np.array(score_list))[3]
    loc_max=loc_list[max_ind[1]]

    cv2.rectangle(img_p_o,(loc_max[0][0],loc_max[0][1]),(loc_max[0][0]+size_list[max_ind[1]][1],loc_max[0][1]+size_list[max_ind[1]][0]),(0,0,255),2)

    cv2.imshow("cursor"+str(i_c),img_p_o)
    dir_str_2="/Users/yangyunchen/Dropbox/Python/proj1_cse573/task3/images/pos/poswrite/cursor"+str(i_c)+".png"
    cv2.imwrite(dir_str_2,img_p_o)

cv2.waitKey(0)
cv2.destroyAllWindows()