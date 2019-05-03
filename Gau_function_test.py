import cv2
import numpy
from numpy import *
import numpy as np
import math

def gaus_kernel(sigma,pixel_num):
    Gau=zeros((pixel_num,pixel_num))
    hw=1/2*(pixel_num-1)
    ##hw=half width, being calculated to adjust the center position
    ##
    for x_g in range(0,pixel_num):
        for y_g in range(0,pixel_num):
            Gau[x_g,y_g]=(1/(2*math.pi*sigma*sigma))*exp(-((x_g-hw)*(x_g-hw)+(y_g-hw)*(y_g-hw))/(2*sigma*sigma))
    sum_Gau=sum(Gau)
    for x_g in range(0,pixel_num):
        for y_g in range(0,pixel_num):
            Gau[x_g,y_g]=Gau[x_g,y_g]/sum_Gau
    return Gau

def image_filter(img_input,filter_kernel):
    
    size_img_input=img_input.shape
    kernel_size=filter_kernel.shape
    rgb_gray=len(size_img_input)
    if rgb_gray==3: 
        new_img=zeros((size_img_input[0],size_img_input[1],size_img_input[2]))
        rgb_channel=size_img_input[2]
        for c in range(0,rgb_channel):
            img_ex=img_input[:,:,c]
            for i in range(0,size_img_input[0]):
                for j in range(0,size_img_input[1]):
                    H=zeros((kernel_size[0],kernel_size[1]))
                    G=zeros((kernel_size[0],kernel_size[1]))
                    for n in range (int(-(kernel_size[1]-1)/2)),int((kernel_size[1]+1)/2):
                        for m in range (int(-(kernel_size[1]-1)/2),int((kernel_size[1]+1)/2)):
                            if i+n in range (0,size_img_input[0]) and j+m in range (0,size_img_input[1]):
                                H[1+n,1+m]=img_ex[i+n,j+m]
                            else:
                                H[1+n,1+m]=0
                    for x in range(-1,2):
                        for y in range(-1,2):
                            G[1+x,1+y]=H[1+x,1+y]*filter_kernel[1-x,1-y]
                    new_img[i,j,c]=sum(G)
    else:
        new_img=zeros((size_img_input[0],size_img_input[1]))
        for i in range(0,size_img_input[0]):
            for j in range(0,size_img_input[1]):
                H=zeros((kernel_size[0],kernel_size[1]))
                G=zeros((kernel_size[0],kernel_size[1]))
                for n in range (int(-(kernel_size[1]-1)/2),int((kernel_size[1]+1)/2)):
                    for m in range (int(-(kernel_size[1]-1)/2),int((kernel_size[1]+1)/2)):
                        if i+n in range (0,size_img_input[0]) and j+m in range (0,size_img_input[1]):
                            H[1+n,1+m]=img_input[i+n,j+m]
                        else:
                            H[1+n,1+m]=0
                for x in range(-1,2):
                    for y in range(-1,2):
                        G[1+x,1+y]=H[1+x,1+y]*filter_kernel[1-x,1-y]
                new_img[i,j]=sum(G)
        
    img_new_adjusted=new_img.clip(0, 255)
    img_new=np.rint(img_new_adjusted).astype('uint8')
    return img_new
    
k=gaus_kernel(1,3)
img=cv2.imread("/Users/yangyunchen/Dropbox/Python/task2.jpg")
img_conv=image_filter(img,k)

cv2.imshow('IMG',img)
cv2.imshow('IMG_cov',img_conv)
cv2.waitKey(0)
cv2.destroyAllWindows()







