import cv2
import numpy as np
import copy
import sys

def kmeans_img(image,k):
    img=image.astype('float64')
    l=img.shape[0]
    w=img.shape[1]
    dataset=[]
    locset=[]
    for i_l in range(0,l):
        for i_w in range (0,w):
            info=[img[i_l,i_w,0],img[i_l,i_w,1],img[i_l,i_w,2]]
            loc=[i_l,i_w]
            dataset.append(info)
            locset.append(loc)
    ind_r=np.random.randint(0,len(dataset),k)
    b_ran=[dataset[x][0] for x in ind_r]
    g_ran=[dataset[x][1] for x in ind_r]
    r_ran=[dataset[x][2] for x in ind_r]
    b_tem=[0]*k
    g_tem=[0]*k
    r_tem=[0]*k
    comp=0
    while comp==0:
        cluster_flag=np.zeros([len(dataset),1]).astype(int)
        for n in range(0,len(dataset)):
            dis=[]
            for m in range(0,k):
                d=((dataset[n][0]-b_ran[m])**2+(dataset[n][1]-g_ran[m])**2+(dataset[n][2]-r_ran[m])**2)**0.5
                dis.append(d)
            flag=np.argmin(dis)
            cluster_flag[n]=flag
        for i in range(0,k):
            index=[]
            for j in range(0,len(dataset)):
                if cluster_flag[j]==i:
                    index.append(j)
            b_tem[i],g_tem[i],r_tem[i]=np.mean([dataset[ii] for ii in index],axis=0)
        if np.all([np.all(np.round(b_ran)==np.round(b_tem)),np.all(np.round(g_ran)==np.round(g_tem)),np.all(np.round(r_ran)==np.round(r_tem))]):
            comp=1
        else:
            b_ran=copy.deepcopy(b_tem)
            g_ran=copy.deepcopy(g_tem)
            r_ran=copy.deepcopy(r_tem)
        
    img_new=np.zeros([l,w,3])
    for num in range(0,len(dataset)):
        img_new[locset[num][0],locset[num][1],:]=[b_ran[cluster_flag[num][0]],g_ran[cluster_flag[num][0]],r_ran[cluster_flag[num][0]]]    
    img_new=img_new.astype('uint8')
    
    return img_new

path=sys.path[0]+'\\' 
img_km=cv2.imread(path+"baboon.jpg")
img_3=kmeans_img(img_km,3)
img_5=kmeans_img(img_km,5)
img_10=kmeans_img(img_km,10)
img_20=kmeans_img(img_km,20)

cv2.imwrite(path+'task3_baboon_3.jpg',img_3)
cv2.imwrite(path+'task3_baboon_5.jpg',img_5)
cv2.imwrite(path+'task3_baboon_10.jpg',img_10)
cv2.imwrite(path+'task3_baboon_20.jpg',img_20)  