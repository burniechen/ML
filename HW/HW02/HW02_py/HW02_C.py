import gzip
import numpy as np
from numpy import pi as pi

from data_prepare import *

# -----------Continuous mode-----------

def GetMeanVarOfEachPixel(image_data, index_m):
    tmp = []
    mean_var = []

    for num in range(10):
        for i in index_m[num]:
            tmp.append(image_data[i])
        tmp_mean = np.mean(tmp, axis=0) #axis = 0 → column
        tmp_var = np.var(tmp, axis=0)
        
        tmp_var = np.where(tmp_var==0, 10000, tmp_var)
            
        mean_var.append([tmp_mean, tmp_var])
        tmp = []
    
    return mean_var

def GetGaussianP(x, mean, sigma_2):

    coef = 1 / pow(2 * pi * sigma_2, 0.5)
    normal = -0.5 * (pow(x - mean, 2) / sigma_2)
    
    return np.log(coef) + normal

def GetPosterior(index, label):
    posterior = 0

    prior = np.log(len(train_index[label]) / train_label_num)

    for pixel in range(row * column):
        posterior += GetGaussianP(test_image_data[index][pixel], \
                                   train_mean_var[label][0][pixel], \
                                   train_mean_var[label][1][pixel]) + prior

    return posterior

def GetPredictValue(i):
    posterior_m = []
    posterior_normalize = []

    for label in range(10):
        posterior_m.append(GetPosterior(i, label))

    total = sum(posterior_m)

    for label in range(10):
        normal = posterior_m[label] / total
        print(f'{label} : {normal}')
        posterior_normalize.append(normal)

    ans = np.argmin(posterior_normalize)
    print(f'Prediction: {test_label_data[i]}, Ans: {ans}\n')

    return ans

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

# Get mean & variance for Gaussion Distribution & Prior
train_index = GetIndexOfEachLabel(train_label_data)
train_mean_var = GetMeanVarOfEachPixel(train_image_data, train_index)

data_num = 10

# Continuous mode
error = 0
for i in range(data_num):
    predict_val = GetPredictValue(i)
    if predict_val != test_label_data[i]:
        error += 1
print(f'Error rate : {error / data_num}')