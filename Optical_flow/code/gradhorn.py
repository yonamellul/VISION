import numpy as np
import matplotlib.pyplot as plt
from middlebury import *



#def gradhorn(I1,I2):
#  Ix = np.zeros(I1.shape)
#  Iy = np.zeros(I1.shape)
#  It = np.zeros(I1.shape)
#
#  for i in range(Ix.shape[0]-1):
#    for j in range(Ix.shape[1]-1):
#      Ix[i,j] = 1/4 * (I1[i,j+1] - I1[i,j] + I1[i+1,j+1] - I1[i+1,j] + I2[i,j+1] - I2[i,j] + I2[i+1,j+1] - I2[i+1,j])
#      Iy[i,j] = 1/4 * (I1[i+1,j] - I1[i,j] + I1[i+1,j+1] - I1[i,j+1] + I2[i+1,j] - I2[i,j] + I2[i+1,j+1] - I2[i,j+1])
#      It[i,j] = 1/4 * (I2[i,j] - I1[i,j] + I2[i+1,j] - I1[i+1,j] + I2[i,j+1] - I1[i,j+1] + I2[i+1,j+1] - I1[i+1,j+1])

# return Ix, Iy, It


def gradhorn(I1,I2):
    dx=np.empty((I1.shape[0],I1.shape[1]))
    dy=np.empty((I1.shape[0],I1.shape[1]))
    dz=np.empty((I1.shape[0],I1.shape[1]))

    dx[:,-1]=0.0
    dx[-1,:]=0.0
    dy[:,-1]=0.0
    dy[-1,:]=0.0
    dz[:,-1]=0.0
    dz[-1,:]=0.0
    
    dx[:-1,:-1]= I1[:-1,1:] #i j+1
    dx[:-1,:-1]-=I1[:-1,:-1] #i j
    dx[:-1,:-1]+= I1[1:,1:]  #i+1 j+1
    dx[:-1,:-1]-=I1[1:,:-1]  #i+1 j
    
    dx[:-1,:-1]+= I2[:-1,1:]  #i j+1
    dx[:-1,:-1]-=I2[:-1,:-1] #i j
    dx[:-1,:-1]+= I2[1:,1:]  #i+1 j+1
    dx[:-1,:-1]-=I2[1:,:-1]  #i+1 j

    dy[:-1,:-1]= I1[1:,:-1] #i j+1
    dy[:-1,:-1]-=I1[:-1,:-1] #i j
    dy[:-1,:-1]+= I1[1:,1:]  #i+1 j+1
    dy[:-1,:-1]-=I1[:-1,1:]  #i+1 j
    
    dy[:-1,:-1]+= I2[1:,:-1]  #i j+1
    dy[:-1,:-1]-=I2[:-1,:-1] #i j
    dy[:-1,:-1]+= I2[1:,1:]  #i+1 j+1
    dy[:-1,:-1]-=I2[:-1,1:]  #i+1 j

    dz[:-1,:-1]= I2[:-1,:-1] #i j+1
    dz[:-1,:-1]-= I1[:-1,:-1] #i j
    dz[:-1,:-1]+= I2[1:,:-1]  #i+1 j+1
    dz[:-1,:-1]-=I1[1:,:-1]  #i+1 j
    
    dz[:-1,:-1]+= I2[:-1,1:]  #i j+1
    dz[:-1,:-1]-=I1[:-1,1:] #i j
    dz[:-1,:-1]+= I2[1:,1:]  #i+1 j+1
    dz[:-1,:-1]-=I1[1:,1:]  #i+1 j

    return dx*0.25,dy*0.25,dz*0.25

if __name__ == "__main__":
    I1 = plt.imread("../data/yosemite/yos9.png")
    I2 = plt.imread("../data/yosemite/yos10.png")
    #I1 = plt.imread("../data/square/square9.png")
    #I2 = plt.imread("../data/square/square10.png")
    Ix,Iy,It = gradhorn(I1,I2)
    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.imshow(Ix)
    plt.title('I_x')
    fig.add_subplot(2, 2, 2)
    plt.imshow(Iy)
    plt.title('I_y')
    fig.add_subplot(2, 2, 3)
    plt.imshow(It)
    plt.title('I_t')
    fig.add_subplot(2, 2, 4)
    plt.imshow(computeColor(np.dstack((Ix, Iy))))
    plt.title('computeColor')
    plt.show()