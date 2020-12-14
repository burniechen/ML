import numpy as np
import cv2
import imageio
import random
from scipy.spatial.distance import pdist, squareform
from numba import jit
from scipy import linalg

@jit
def GetMeanIndex(k, mode=False):
    mean_index = []

    # Random k-means
    for i in range(k):
        index = random.randint(0, row*col-1)
        while index in mean_index:
            index = random.randint(0, row*col-1)
        mean_index.append(index)

    if mode:
        # k-means++
        mean_index[0] = random.randint(0, row*col-1)

        for _k in range(1, k):
            distance = np.zeros((row*col, ))
            classify = np.zeros((row*col, ))
            closest = np.zeros((k, ))
            for p in range(row*col):
                for i in range(_k):
                    closest[i] = np.linalg.norm(GramMatrix[p] - GramMatrix[mean_index[i]])
                classify[p] = closest.argmin()
                
                distance[p] = np.linalg.norm(GramMatrix[p] - GramMatrix[mean_index[int(classify[p])]])
            
            dis_sum = np.sum(distance)
            val = random.uniform(0, dis_sum)
            count = 0
            while val < 0:
                val -= distance[count]
                count += 1
            mean_index[_k] = count

    return mean_index

@jit
def Estep(GramMatrix, mean, k):
    classify = np.zeros((row*col, ))
    distance = np.zeros((k, ))
    for i in range(row*col):
        for j in range(k):
            distance[j] = np.linalg.norm(GramMatrix[i] - mean[j])
        classify[i] = distance.argmin()
        
    return classify

@jit
def Mstep(GramMatrix, classify, k):
    new_mean = np.zeros((k, row*col))
    for i in range(k):
        cluster = GramMatrix[classify == i]
        new_mean[i] = cluster.sum(axis = 0)
        if len(cluster) > 0:
            new_mean[i] /= len(cluster)
            
    return new_mean

path = "ML_HW06/image1.png"
img = cv2.imread(path)

row, col, channel = img.shape

# Spatial info
SpatialInfo = [[i, j] for i in range(row) for j in range(row)]
SpatialInfo = np.array(SpatialInfo)

# Spatial Kernel
SpatialKernel = (-1/10000 * pdist(SpatialInfo, 'sqeuclidean'))

#Img info
ImgInfo = img.reshape(row*col, channel)

#Img Kernel
ImgKernel = (-1/10000 * pdist(ImgInfo, 'sqeuclidean'))

NewKernel = np.exp(SpatialKernel + ImgKernel)

# Gram matrix
GramMatrix = squareform(NewKernel)

k = 5

# Mean index
mean_index = GetMeanIndex(k, mode=False)
mean = GramMatrix[mean_index]

gif = np.zeros((1, row, col))

row_sum = np.sum(GramMatrix, axis=1)
Degree = np.diag(row_sum)

UnnormalizedLap = Degree - GramMatrix

InvSqrtDegree = np.linalg.inv(np.sqrt(Degree))
NormalizedLap = InvSqrtDegree @ UnnormalizedLap @ InvSqrtDegree

eig = linalg.eigh(NormalizedLap)

U = eig[1][:, :k]
norm_U = np.linalg.norm(U, axis=1)

T = U
T = np.array([U[i]/norm_U[i] for i in range(len(U))])

Tkernel = np.exp((-1/100000 * pdist(T, 'sqeuclidean')))
T_GramMatrix = squareform(Tkernel)
T_mean = T_GramMatrix[mean_index]

# Update k-means
flag = True
while flag:
    classify = Estep(T_GramMatrix, T_mean, k)
    new_mean = Mstep(T_GramMatrix, classify, k)

    gif = np.vstack((gif, classify.reshape(1, row, col)))
    print(np.linalg.norm(new_mean - T_mean))

    if np.linalg.norm(new_mean - T_mean) < 1:
        flag = False
    T_mean = new_mean

ColorMap = [
    [  0,   0,   0],
    [255,   0,   0],
    [255, 255,   0],
    [  0, 255,   0],
    [ 65, 255, 225],
    [160,  32, 240]
]

colorful_gif = []
for count in range(gif.shape[0]):
    colorful_gif.append([])
    for i in range(gif.shape[1]):
        for j in range(gif.shape[2]):
            color = ColorMap[int(gif[count][i][j])]
            colorful_gif[count].append(color)

colorful_gif = np.array(colorful_gif).reshape(gif.shape[0], row, col, 3)

imageio.mimsave("sp.gif", colorful_gif, 'GIF', duration=0.5)