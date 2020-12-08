import numpy as np
import gzip
from data_prepare import *

train_image = 'train-images-idx3-ubyte.gz'
train_label = 'train-labels-idx1-ubyte.gz'

test_image = 't10k-images-idx3-ubyte.gz'
test_label = 't10k-labels-idx1-ubyte.gz'

# Train
f = gzip.open(train_image,'rb')
data = GetImageData(f)
train_image_data, train_image_num, row, column = data

f = gzip.open(train_label,'rb')
data = GetLabelData(f)
train_label_data, train_label_num = data

# Test
f = gzip.open(test_image,'rb')
data = GetImageData(f)
test_image_data, test_image_num, row, column = data

f = gzip.open(test_label,'rb')
data = GetLabelData(f)
test_label_data, test_label_num = data

train_grey_data = np.where(train_image_data < 128, 0, 1)

w = np.zeros((10, train_image_num)) # w[10][60000]
lam = np.zeros((10, )) # lam[10]
prob = np.zeros((10, row*column)) # prob[10][784]

train_index = GetIndexOfEachLabel(train_label_data)

for i in range(len(train_index)):
    lam[i] = len(train_index[i])
lam = lam/train_image_num

for i in range(10):
    w[i] += lam[i]

peudoP = 0.05
prob += peudoP

# E-step
for n in range(60000):
    print(f'============ n: {n}=================')
    totalP = 0
    for label in range(10):
        P = lam[label]
        for pixel in range(row*column):
            P *= (prob[label][pixel] ** train_grey_data[n][pixel]) * ((1 - prob[label][pixel]) ** (1 - train_grey_data[n][pixel]))
            w[label][n] = P
            totalP += P
            print(totalP, P)
        w[label][n] /= totalP