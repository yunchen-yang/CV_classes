#import cv2
from numpy import *
import numpy as np
import math
##
#define image filter function

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
                                H[1+n,1+m]=img_input[i+n,j+m]
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

##
#define function to generate gaussian kernel

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
                    sampled_img[x_adj,y_adj,colorchannel]=np.mean(image_original[ratio*x_adj:ratio*(x_adj+1),ratio*y_adj:ratio*(y_adj+1),colorchannel])
    else:
        sampled_img=zeros((length_adj,width_adj))
        for x_adj in range(0,length_adj):
                for y_adj in range(0,width_adj):
                    sampled_img[x_adj,y_adj]=np.mean(image_original[ratio*x_adj:ratio*(x_adj+1),ratio*y_adj:ratio*(y_adj+1)])

    sampled_img_output=np.rint(sampled_img).astype('uint8')
    return sampled_img_output
##
##define octave function
def octave_array(img_base,scale_number,sigma_initial):
    size_base=img_base.shape
    octave_mat=zeros((size_base[0],size_base[1],scale_number))   
    for scale_count in range(0,scale_number):
        sigma_octave=sigma_initial*(pow(pow(2,1/2),scale_count))
        kernel_oc=gaus_kernel(sigma_octave,3)
        octave_mat[:,:,scale_count]=image_filter(img_base,kernel_oc)
    
    octave_output=np.rint(octave_mat).astype('uint8')
    return octave_output

    
##define DoG converting funtion
def DoG_convert(octave_tower):
    size_tower=octave_tower.shape
    DoG=zeros((size_tower[0],size_tower[1],size_tower[2]-1))
    for tower_n in range(0,size_tower[2]-1):
        DoG[:,:,tower_n]=octave_tower[:,:,tower_n+1].astype(float)-octave_tower[:,:,tower_n].astype(float)
    
    return DoG


##define finding local max/min function
def local_maxmin(DoG):
    size_dog=DoG.shape
    coor_list=[] #coordinator lists of keypoint candidates
    for n_doglayer in range(1,size_dog[2]-1):
        for x_dog in range(1,size_dog[0]-1):
            for y_dog in range(1,size_dog[1]-1):
                
                local_max=max(DoG[x_dog-1,y_dog-1,n_doglayer-1],DoG[x_dog-1,y_dog,n_doglayer-1],DoG[x_dog-1,y_dog+1,n_doglayer-1],
                DoG[x_dog+1,y_dog-1,n_doglayer-1],DoG[x_dog+1,y_dog,n_doglayer-1],DoG[x_dog+1,y_dog+1,n_doglayer-1],
                DoG[x_dog,y_dog-1,n_doglayer-1],DoG[x_dog,y_dog,n_doglayer-1],DoG[x_dog,y_dog+1,n_doglayer-1],
                DoG[x_dog-1,y_dog-1,n_doglayer+1],DoG[x_dog-1,y_dog,n_doglayer+1],DoG[x_dog-1,y_dog+1,n_doglayer+1],
                DoG[x_dog+1,y_dog-1,n_doglayer+1],DoG[x_dog+1,y_dog,n_doglayer+1],DoG[x_dog+1,y_dog+1,n_doglayer+1],
                DoG[x_dog,y_dog-1,n_doglayer+1],DoG[x_dog,y_dog,n_doglayer+1],DoG[x_dog,y_dog+1,n_doglayer+1],
                DoG[x_dog-1,y_dog-1,n_doglayer],DoG[x_dog-1,y_dog,n_doglayer],DoG[x_dog-1,y_dog+1,n_doglayer],
                DoG[x_dog+1,y_dog-1,n_doglayer],DoG[x_dog+1,y_dog,n_doglayer],DoG[x_dog+1,y_dog+1,n_doglayer],
                DoG[x_dog,y_dog-1,n_doglayer],DoG[x_dog,y_dog+1,n_doglayer])
                
                local_min=min(DoG[x_dog-1,y_dog-1,n_doglayer-1],DoG[x_dog-1,y_dog,n_doglayer-1],DoG[x_dog-1,y_dog+1,n_doglayer-1],
                DoG[x_dog+1,y_dog-1,n_doglayer-1],DoG[x_dog+1,y_dog,n_doglayer-1],DoG[x_dog+1,y_dog+1,n_doglayer-1],
                DoG[x_dog,y_dog-1,n_doglayer-1],DoG[x_dog,y_dog,n_doglayer-1],DoG[x_dog,y_dog+1,n_doglayer-1],
                DoG[x_dog-1,y_dog-1,n_doglayer+1],DoG[x_dog-1,y_dog,n_doglayer+1],DoG[x_dog-1,y_dog+1,n_doglayer+1],
                DoG[x_dog+1,y_dog-1,n_doglayer+1],DoG[x_dog+1,y_dog,n_doglayer+1],DoG[x_dog+1,y_dog+1,n_doglayer+1],
                DoG[x_dog,y_dog-1,n_doglayer+1],DoG[x_dog,y_dog,n_doglayer+1],DoG[x_dog,y_dog+1,n_doglayer+1],
                DoG[x_dog-1,y_dog-1,n_doglayer],DoG[x_dog-1,y_dog,n_doglayer],DoG[x_dog-1,y_dog+1,n_doglayer],
                DoG[x_dog+1,y_dog-1,n_doglayer],DoG[x_dog+1,y_dog,n_doglayer],DoG[x_dog+1,y_dog+1,n_doglayer],
                DoG[x_dog,y_dog-1,n_doglayer],DoG[x_dog,y_dog+1,n_doglayer])
                
                if DoG[x_dog,y_dog,n_doglayer]>local_max or DoG[x_dog,y_dog,n_doglayer]<local_min:
                    coor_list.append([x_dog,y_dog,n_doglayer])
                else:
                    continue
    return coor_list
                    
     
             
img=cv2.imread("C:/Users/eris9/Dropbox/Pythontask2.jpg",0)
#prepare matrix of different levels of octaves
##

##build octave
octave_1=octave_array(img,5,pow(2,1/2)/2)
img_base_2=sampling(octave_1[:,:,2],2)
octave_2=octave_array(img_base_2,5,pow(2,1/2))
img_base_3=sampling(octave_2[:,:,2],2)
octave_3=octave_array(img_base_3,5,pow(2,1/2)*2)
img_base_4=sampling(octave_3[:,:,2],2)
octave_4=octave_array(img_base_4,5,pow(2,1/2)*4)

##calculate DoG
DoG_1=DoG_convert(octave_1)
DoG_2=DoG_convert(octave_2)
DoG_3=DoG_convert(octave_3)
DoG_4=DoG_convert(octave_4)

##
#finding out the local max/min
coor_1=local_maxmin(DoG_1)
coor_2=local_maxmin(DoG_2)
coor_3=local_maxmin(DoG_3)
coor_4=local_maxmin(DoG_4)

cv2.imshow('OC1_1',octave_1[:,:,0])
cv2.imshow('OC1_2',octave_1[:,:,1])
cv2.imshow('OC1_3',octave_1[:,:,2])
cv2.imshow('OC1_4',octave_1[:,:,3])
cv2.imshow('OC1_5',octave_1[:,:,4])

cv2.imshow('DoG1_1',DoG_1[:,:,0])
cv2.imshow('DoG1_2',DoG_1[:,:,1])
cv2.imshow('DoG1_3',DoG_1[:,:,2])
cv2.imshow('DoG1_4',DoG_1[:,:,3])

cv2.imshow('IMG',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




















#cv2.imshow('SAM',sampling(img,2))
#cv2.waitKey(0)
#cv2.destroyAllWindows()
            
        
        
        
        
        
        
        
        
        
        
        