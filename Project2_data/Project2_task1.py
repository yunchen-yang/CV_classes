import cv2
import numpy as np
import copy
import sys

path=sys.path[0]+'/'
img1=cv2.imread(path+"mountain1.jpg")
img2=cv2.imread(path+"mountain2.jpg")

img1_g=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_g=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
Kp1, des1=sift.detectAndCompute(img1_g,None)
Kp2, des2=sift.detectAndCompute(img2_g,None)
img1_k=copy.deepcopy(img1)
img2_k=copy.deepcopy(img2)
cv2.drawKeypoints(img1,Kp1,img1_k,color=(0,0,255))
cv2.drawKeypoints(img2,Kp2,img2_k,color=(0,0,255))

bf = cv2.BFMatcher()
matching = bf.knnMatch(des1,des2, k=2)
good_match=[]

for m,n in matching:
    if m.distance<0.75*n.distance:
        good_match.append(m)

img_match_knn=cv2.drawMatches(img1,Kp1,img2,Kp2,good_match,None,(0,255,255),None,None,2)
KpL1=[]
KpL2=[]
for i in range(0,len(good_match)):
    KpL1.append(Kp1[good_match[i].queryIdx].pt)
    KpL2.append(Kp2[good_match[i].trainIdx].pt)

KpLpt1=np.float32(KpL1).reshape(-1,1,2)
KpLpt2=np.float32(KpL2).reshape(-1,1,2)

Matrix_H, mask=cv2.findHomography(KpLpt2,KpLpt1,cv2.RANSAC,5)
print(Matrix_H)
mask_list=mask.ravel().tolist()
count=0
index=[]
while count<10:
    num=np.random.randint(0,len(good_match))
    if num in index:
        continue
    else:
        if mask_list[num]==1:
            count=count+1
            index.append(num)

img_homo=cv2.drawMatches(img1,Kp1,img2,Kp2,good_match,None,(0,0,255),None,mask_list,2)
img_homo_rand=cv2.drawMatches(img1,Kp1,img2,Kp2,[good_match[i] for i in index],None,(0,0,255),None,[mask_list[i] for i in index],2)
img_pano=np.zeros((img1.shape[0],img1.shape[1]+img2.shape[1],3),np.uint8)
for chan in range(0,3):
    pano=cv2.warpPerspective(img2[:,:,chan],Matrix_H,(img1.shape[1]+img2.shape[1],img1.shape[0]))
    pano[0:img1.shape[0],0:img1.shape[1]]=img1[:,:,chan]
    img_pano[:,:,chan]=pano

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("img1_k",img1_k)
cv2.imshow("img2_k",img2_k) 
cv2.imshow("Matching_knn",img_match_knn)
cv2.imshow("Matching_homo",img_homo)
cv2.imshow("Matching_homo_rand",img_homo_rand)
cv2.imshow("img_pano",img_pano)
