import cv2
import sys
import matplotlib.pyplot as plt

path=sys.path[0]+'/' 
img=cv2.imread(path+"original_imgs/segment.jpg",0)

img_line=tuple(img.reshape(1,-1).astype('int')[0])
hist={}
for n in img_line:
    hist[n]=hist.get(n,0)+1
del hist[0]
l=len(img_line)
for n in hist.keys():
    plt.bar(n,(hist[n]/l),width=1,color='b')
plt.xlabel('Intensity')
plt.ylabel('Percentage')
plt.show()