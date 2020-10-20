import sys
import numpy as np
from math import factorial as fac

def BinomialCoef(n, m):
    return fac(n) / (fac(m) * fac(n - m))

def GetSuccessCount(index):
    N = len(data[index])
    n = 0
    for i in range(N):
        if int(data[index][i]) == 0:
            n += 1

    m = N - n
    
    return [N, m]

f = open("testfile.txt")
allData = f.read()
f.close()

data = allData.split()

a = int(sys.argv[1])
b = int(sys.argv[2])
for i in range(len(data)):
    # N: total, m: success
    N, m = GetSuccessCount(i)
    p = m / N
    likelihood = BinomialCoef(N, m) * pow(p, m) * pow((1 - p), N - m)
    print(f'case {i+1}: {data[i]}')
    print(f'Likelihood: {likelihood}')
    print(f'Beta prior: a = {a}, b = {b}')
    a += m
    b += N - m
    print(f'Beta posterior: a = {a}, b = {b}\n')