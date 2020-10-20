import gzip
import numpy as np
from numpy import pi as pi

def GetImageData(f):
    # magic number
    f.read(4) 
    
    # number of images
    num = f.read(4)
    num = int.from_bytes(num, byteorder='big') #60000

    row = f.read(4)
    row = int.from_bytes(row, byteorder='big') #28

    column = f.read(4)
    column = int.from_bytes(column, byteorder='big') #28

    buf = f.read(row * column * num)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num, row * column)
    
    return [data, num, row, column]

def GetLabelData(f):
    # magic number
    f.read(4) 

    # number of items
    num = f.read(4)
    num = int.from_bytes(num, byteorder='big') #60000

    buf = f.read(num)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num)
    
    return [data, num]

def GetIndexOfEachLabel(label_data):
    index = 0
    index_m = []
    for num in range(10):
        index_m.append([])
        for label in label_data:
            if label == num:
                index_m[num].append(index)
            index += 1
        index = 0
        
    return index_m

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
        # print(f'{label} : {normal}')
        posterior_normalize.append(normal)
        
    return np.argmin(posterior_normalize)

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
            print(image[i][j], " ", end = "")
        print('\n')

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

train_index = GetIndexOfEachLabel(train_label_data)
test_index = GetIndexOfEachLabel(test_label_data)

# Tally the frequency of the values of each pixel into 32 bins
test_image_classify = np.trunc(test_image_data / 8).astype(int)

prob_each_bin = []
for label in range(10):
    prob_each_bin.append(GetProbOfEachBin(label))

data_num = 10000

# Discrete mode
error = 0
for i in range(data_num):
    predict_val = GetPredictValueDC(i)
    if predict_val != test_label_data[i]:
        error += 1
print(f'Discrete mode \nError rate : {error / data_num}')

for label in range(10):
    print(f'{label}：')
    PrintClassifyImage(label)
    print('\n\n\n')