import cv2
from numpy import *
import numpy as np
import math
##
#define image filter function

def image_filter(img_input,filter_kernel):
    
    size_img_input=img_input.shape
    kernel_size=filter_kernel.shape
    decenter=int((kernel_size[1]-1)/2)
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
                    for n in range(-decenter,decenter+1):
                        for m in range(-decenter,decenter+1):
                            if i+n in range(0,size_img_input[0]) and j+m in range(0,size_img_input[1]):
                                H[decenter+n,decenter+m]=img_ex[i+n,j+m]
                            else:
                                H[decenter+n,decenter+m]=0
                    for x in range(-decenter,decenter+1):
                        for y in range(-decenter,decenter+1):
                            G[decenter+x,decenter+y]=H[decenter+x,decenter+y]*filter_kernel[decenter-x,decenter-y]
                    new_img[i,j,1-(c-1)]=sum(G)
    else:
        new_img=zeros((size_img_input[0],size_img_input[1]))
        for i in range(0,size_img_input[0]):
                for j in range(0,size_img_input[1]):
                    H=zeros((kernel_size[0],kernel_size[1]))
                    G=zeros((kernel_size[0],kernel_size[1]))
                    for n in range(-decenter,decenter+1):
                        for m in range(-decenter,decenter+1):
                            if i+n in range(0,size_img_input[0]) and j+m in range(0,size_img_input[1]):
                                H[decenter+n,decenter+m]=img_input[i+n,j+m]
                            else:
                                H[decenter+n,decenter+m]=0
                    for x in range(-decenter,decenter+1):
                        for y in range(-decenter,decenter+1):
                            G[decenter+x,decenter+y]=H[decenter+x,decenter+y]*filter_kernel[decenter-x,decenter-y]
                    new_img[i,j]=sum(G)
        
    img_new_adjusted=new_img.clip(0, 255)
    img_new=np.rint(img_new_adjusted).astype('uint8')
    return img_new

##
#define function to generate gaussian kernel

def gaus_kernel(sigma,pixel_num):
    Gau=zeros((pixel_num,pixel_num))
    hw=1/2*(pixel_num-1)
    ##hw=half width, being calculated to adjust the center position
    ##
    for x_g in range(0,pixel_num):
        for y_g in range(0,pixel_num):
            Gau[x_g,y_g]=(1/(2*math.pi*sigma*sigma))*exp(-((x_g-hw)*(x_g-hw)*sigma*sigma+(y_g-hw)*(y_g-hw)*sigma*sigma)/(2*sigma*sigma))
    sum_Gau=sum(Gau)
    for x_g in range(0,pixel_num):
        for y_g in range(0,pixel_num):
            Gau[x_g,y_g]=Gau[x_g,y_g]/sum_Gau
    return Gau


##
#load images
##
img=cv2.imread("/Users/yangyunchen/Dropbox/Python/task1.png")
cv2.imshow('IMG',img)
##
#define kernel matrix for as image filter
sobelv=array([[1,0,-1],[2,0,-2],[1,0,-1]])
sobelh=transpose(sobelv)
Gau_77=gaus_kernel(sqrt(2),7)

##
#image processing
img_g=image_filter(img,Gau_77)
edge_vertical=image_filter(img_g,sobelv)
edge_horizontal=image_filter(img_g,sobelh)
##
#present the processed images
cv2.imshow('IMG_edge_vertical',edge_vertical)
cv2.imwrite('/Users/yangyunchen/Dropbox/Python/IMG_edge_vertical.png',edge_vertical)
cv2.imshow('IMG_edge_horizontal',edge_horizontal)
cv2.imwrite('/Users/yangyunchen/Dropbox/Python/IMG_edge_horizontal.png',edge_horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()