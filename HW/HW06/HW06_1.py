import numpy as np
import cv2
import imageio
import random
from scipy.spatial.distance import pdist, squareform
from numba import jit

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

path = "ML_HW06/image2.png"
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

k = 2

# Mean index
mean_index = GetMeanIndex(k, mode=True)
mean = GramMatrix[mean_index]

gif = np.zeros((1, row, col))

# Update k-means
flag = True
while flag:
    classify = Estep(GramMatrix, mean, k)
    new_mean = Mstep(GramMatrix, classify, k)

    gif = np.vstack((gif, classify.reshape(1, row, col)))
    print(np.linalg.norm(new_mean - mean))

    if np.linalg.norm(new_mean - mean) < 1:
        flag = False
    mean = new_mean

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

imageio.mimsave("kmeans.gif", colorful_gif, 'GIF', duration=0.5)