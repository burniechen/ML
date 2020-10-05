import numpy as np
import matplotlib.pyplot as plt
import sys

from rLSE import *
from newton_method import *
from LSEerror_func import *

def PlotLSE(data, coef_array):
    X_data = []
    Y_data = []
    for ele in data:
        X_data.append(ele[0])
        Y_data.append(ele[1])

    x = np.linspace(min(X_data)-1, max(X_data)+1, 100)
    y = []
    for num in x:
        val = 0
        for i in range(0, len(coef_array)):
            val += pow(num, i) * coef_array[i][0]
        y.append(val)

    plt.figure(figsize=(10, 8))
    plt.plot(X_data, Y_data, 'ro')
    plt.plot(x, y)
    plt.show()

def PrintInfo(data, x, type_name):
    type_str = type_name + ":\nFitting line: "
    for i in range(len(x), 0, -1):
        tmp = "(" + str(x[i-1][0]) + ") "
        type_str += tmp
        if i-1 != 0:
            type_str += "X^" + str(i-1) + " + "
    print(f'{type_str}\nTotal Error: {LSEerror(data, x)}')

filename = sys.argv[1]
power = int(sys.argv[2])
lam = int(sys.argv[3])

# load data from txt and "allData" is string
f = open(filename)
allData = f.read()
f.close()

# transform data type from string to list
data = []
for ele in allData.split():
    ele = ele.split(",")
    data.append(ele)

for i in range(0, len(data)):
    for j in range(0, len(data[0])):
        data[i][j] = float(data[i][j])

x1 = rLSE(data, power, lam)
PrintInfo(data, x1, "LSE")

print()

x2 = NewtonMethod(data, power)
PrintInfo(data, x2, "Newton's Method")

PlotLSE(data, x1)
PlotLSE(data, x2)