import gzip
import numpy as np
from numpy import pi as pi

from data_prepare import *

# -----------Discrete mode-----------

def GetProbOfEachBin(label):
    tmp = []
    count = []
    for i in test_index[label]:
        tmp.append(test_image_classify[i])

    tmp = np.array(tmp).transpose()
    
    tmp_c = []
    for b in range(32):
        for pixel in range(784):
            tmp_c.append((tmp[pixel] == b).sum())
        count.append(tmp_c)
        tmp_c = []

    count = np.array(count).transpose()
    count = np.where(count == 0, 0.0001, count)
    prob = count / len(test_index[label])
    
    return prob

def GetPosteriorDC(index, label):
    posterior = 0

    prior = np.log(len(train_index[label]) / train_label_num)
    
    pixel = 0
    for b in test_image_classify[index]:
        posterior += np.log(prob_each_bin[label][pixel][b]) + prior
        pixel += 1

    return posterior

def GetPredictValueDC(i):
    posterior_m = []
    posterior_normalize = []

    for label in range(10):
        posterior_m.append(GetPosteriorDC(i, label))

    total = sum(posterior_m)

    for label in range(10):
        normal = posterior_m[label] / total
        print(f'{label} : {normal}')
        posterior_normalize.append(normal)

    ans = np.argmin(posterior_normalize)
    print(f'Prediction: {test_label_data[i]}, Ans: {ans}\n')

    return ans

def PrintClassifyImage(label):
    image = []
    for pixel in range(784):
        black = np.sum(prob_each_bin[label][pixel][16:32])
        white = np.sum(prob_each_bin[label][pixel][:16])

        if black > white:
            image.append(1)
        else:
            image.append(0)

    image = np.array(image).reshape(28, 28)
    
    for i in range(28):
        for j in range(28):
            print(image[i][j], "", end = "")
        print("")

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

# For Prior
train_index = GetIndexOfEachLabel(train_label_data)
# For Likelihood
test_index = GetIndexOfEachLabel(test_label_data)

# Tally the frequency of the values of each pixel into 32 bins
test_image_classify = np.trunc(test_image_data / 8).astype(int)

prob_each_bin = [] # 10 * 784 * 32
for label in range(10):
    prob_each_bin.append(GetProbOfEachBin(label))

data_num = 10000

# Discrete mode
error = 0
for i in range(data_num):
    predict_val = GetPredictValueDC(i)
    if predict_val != test_label_data[i]:
        error += 1
print(f'Error rate : {error / data_num}\n')

print(f'Imagination of numbers in Bayesian classifier:\n')

for label in range(10):
    print(f'{label}：\n')
    PrintClassifyImage(label)
    print('\n')