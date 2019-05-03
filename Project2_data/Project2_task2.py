import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt

def findingspots(line,img):
    w=img.shape[1]
    lpt1=[]
    lpt2=[]
    for n in range(0,len(line)):
        aa=line[n][0,0]
        bb=line[n][0,1]
        cc=line[n][0,2]
        lpt1.append([0,int(-cc/bb)])
        lpt2.append([w,int(-(aa*w+cc)/bb)])
        
    return lpt1,lpt2
    
img1=cv2.imread("/Users/yangyunchen/Dropbox/Python/Project2_data/tsucuba_left.png")
img2=cv2.imread("/Users/yangyunchen/Dropbox/Python/Project2_data/tsucuba_right.png")

img1_g=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_g=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift=cv2.xfeatures2d.SIFT_create()
Kp1, des1=sift.detectAndCompute(img1_g,None)
Kp2, des2=sift.detectAndCompute(img2_g,None)
img1_k=copy.deepcopy(img1)
img2_k=copy.deepcopy(img2)
cv2.drawKeypoints(img1,Kp1,img1_k,color=(0,0,255))
cv2.drawKeypoints(img2,Kp2,img2_k,color=(0,0,255))

bf=cv2.BFMatcher()
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

KpLpt1=np.float32(KpL1)#.reshape(-1,1,2)
KpLpt2=np.float32(KpL2)#.reshape(-1,1,2)

Matrix_F,mask=cv2.findFundamentalMat(KpLpt1,KpLpt2,cv2.RANSAC,5)
print(Matrix_F)
mask_list=mask.ravel().tolist()
count=0
index=[]
while count<10:
    num=np.random.randint(0,len(good_match))
    if num in index:
        continue
    elif mask_list[num]==0:
        continue
    else:
        count=count+1
        index.append(num)

img_homo=cv2.drawMatches(img1,Kp1,img2,Kp2,good_match,None,(0,0,255),None,mask_list,2)
img_homo_rand=cv2.drawMatches(img1,Kp1,img2,Kp2,[good_match[i] for i in index],None,(0,0,255),None,[mask_list[i] for i in index],2)
kps1=np.int32([KpLpt1[i,:] for i in index])
kps2=np.int32([KpLpt2[i,:] for i in index])

epline1=cv2.computeCorrespondEpilines(kps2.reshape(-1,1,2),2,Matrix_F)
epline2=cv2.computeCorrespondEpilines(kps1.reshape(-1,1,2),1,Matrix_F)
epline1.reshape(-1,3)
epline2.reshape(-1,3)
hpt1_1,hpt1_2=findingspots(epline1,img1_g)
hpt2_1,hpt2_2=findingspots(epline2,img2_g)

img1_ep=copy.deepcopy(img1)
img2_ep=copy.deepcopy(img2)
color=np.random.randint(0,255,30).reshape(10,3)

for num in range(0,10):
    img1_ep=cv2.line(img1_ep,(hpt1_1[num][0],hpt1_1[num][1]),(hpt1_2[num][0],hpt1_2[num][1]),color[num,:].tolist())
    img2_ep=cv2.circle(img2_ep,([KpLpt2[i,:] for i in index][num][0].astype(int),[KpLpt2[i,:] for i in index][num][1].astype(int)),3,color[num,:].tolist(),2)
    img2_ep=cv2.line(img2_ep,(hpt2_1[num][0],hpt2_1[num][1]),(hpt2_2[num][0],hpt2_2[num][1]),color[num,:].tolist())
    img1_ep=cv2.circle(img1_ep,([KpLpt1[i,:] for i in index][num][0].astype(int),[KpLpt1[i,:] for i in index][num][1].astype(int)),3,color[num,:].tolist(),2)

stereo=cv2.StereoBM_create(48,15)
disparity=stereo.compute(img1_g,img2_g)

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("img1_k",img1_k)
cv2.imshow("img2_k",img2_k)
cv2.imshow("Matching_knn",img_match_knn)
cv2.imshow("Matching_knn",img_match_knn)
cv2.imshow("Matching_homo",img_homo)
cv2.imshow("Matching_homo_rand",img_homo_rand)
cv2.imshow("img1_ep",img1_ep)
cv2.imshow("img2_ep",img2_ep)

plt.imshow(disparity,'gray')
plt.show()
