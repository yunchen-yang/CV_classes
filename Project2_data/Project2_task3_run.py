import numpy as np
from matplotlib import pyplot as plt
import sys

path=sys.path[0]+'\\' 
#load dataset
X=np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])

c1=np.array([6.2,3.2])
c2=np.array([6.6,3.7])
c3=np.array([6.5,3.0])
center1=np.array([c1,c2,c3])
d=np.zeros([3,1])
cluster_flag1=np.zeros([len(X),1])
colorbar=['r','b','g']
c_flag=[None]*10
for n in range(0,len(X)):
    d[0]=((X[n,0]-c1[0])**2+(X[n,1]-c1[1])**2)**0.5
    d[1]=((X[n,0]-c2[0])**2+(X[n,1]-c2[1])**2)**0.5
    d[2]=((X[n,0]-c3[0])**2+(X[n,1]-c3[1])**2)**0.5
    flag=np.argmin(d)
    cluster_flag1[n]=flag+1
    c_flag[n]=colorbar[flag]

plt.scatter(X[:,0],X[:,1],c='',edgecolors=c_flag,marker='^')
plt.savefig(path+'task3_iter1_a.jpg')

index1=[]
index2=[]
index3=[]
for nn in range(0,len(X)):
    if cluster_flag1[nn]==1:
        index1.append(nn)
    elif cluster_flag1[nn]==2:
        index2.append(nn)
    else:
        index3.append(nn)

cc1=np.mean([X[i,:] for i in index1],axis=0)
cc2=np.mean([X[i,:] for i in index2],axis=0)
cc3=np.mean([X[i,:] for i in index3],axis=0)
center2=np.array([cc1,cc2,cc3])
plt.figure()
plt.scatter(X[:,0],X[:,1],c='',edgecolors=c_flag,marker='^')


for i2 in range(0,len(colorbar)):
    plt.scatter(center2[i2,0],center2[i2,1],c=colorbar[i2],marker='o')
plt.savefig(path+'task3_iter1_b.jpg')
cluster_flag2=np.zeros([len(X),1])

for n in range(0,len(X)):
    d[0]=((X[n,0]-cc1[0])**2+(X[n,1]-cc1[1])**2)**0.5
    d[1]=((X[n,0]-cc2[0])**2+(X[n,1]-cc2[1])**2)**0.5
    d[2]=((X[n,0]-cc3[0])**2+(X[n,1]-cc3[1])**2)**0.5
    flag=np.argmin(d)
    cluster_flag2[n]=flag+1
    c_flag[n]=colorbar[flag]

plt.figure()
plt.scatter(X[:,0],X[:,1],c='',edgecolors=c_flag,marker='^')

index1=[]
index2=[]
index3=[]
for nnn in range(0,len(X)):
    if cluster_flag2[nnn]==1:
        index1.append(nnn)
    elif cluster_flag2[nnn]==2:
        index2.append(nnn)
    else:
        index3.append(nnn)

ccc1=np.mean([X[i,:] for i in index1],axis=0)
ccc2=np.mean([X[i,:] for i in index2],axis=0)
ccc3=np.mean([X[i,:] for i in index3],axis=0)
center3=np.array([ccc1,ccc2,ccc3])
plt.figure()
plt.scatter(X[:,0],X[:,1],c='',edgecolors=c_flag,marker='^')
plt.savefig(path+'task3_iter2_a.jpg')

for i3 in range(0,len(colorbar)):
    plt.scatter(center3[i3,0],center3[i3,1],c=colorbar[i3],marker='o')
plt.savefig(path+'task3_iter2_b.jpg')


