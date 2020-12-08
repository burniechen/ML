import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from libsvm.svmutil import *
from libsvm.svm import *
import csv

def ReadCSV(filename):
    with open(filename, newline='') as csvfile:
        row = csv.reader(csvfile)
        data_list = [[float(ele) for ele in array] for array in row]
    
    data = np.array(data_list)
    row, col = data.shape
    return [data, row, col]

@jit
def LinearKernel(Xa, Xb):
    Xa = np.array(Xa)
    Xb = np.array(Xb)
    return Xa.T @ Xb

@jit
def RBFkernel(Xa, Xb):
    gamma = 1/784
    difference = np.array([Xa[i] - Xb[i] for i in range(len(Xa))])
    return np.exp(-1 * gamma * (difference.T @ difference))

@jit
def GetDataByKernel(X_train):
    length = X_train.shape[0]
    output = []
    for i in range(length):
        tmp = []
        tmp.append(i + 1)
        for j in range(length):
            val = LinearKernel(X_train[i], X_train[j]) + RBFkernel(X_train[i], X_train[j])
            tmp.append(val)
        output.append(tmp)
    return np.array(output)

X_train, X_train_row, X_train_col = ReadCSV("X_train.csv")
Y_train, Y_train_row, Y_train_col = ReadCSV("Y_train.csv")
Y_train = Y_train.reshape(5000,)

X_test, X_test_row, X_test_col = ReadCSV("X_test.csv")
Y_test, Y_test_row, Y_test_col = ReadCSV("Y_test.csv")
Y_test = Y_test.reshape(2500, )

new_train = GetDataByKernel(X_train)
new_test = GetDataByKernel(X_test)

model = svm_train(Y_train, new_train, '-t 4')
label, acc, val = svm_predict(Y_test, new_test, model)