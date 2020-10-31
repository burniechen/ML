import numpy as np

from data_generator import UnivariateGaussian

mean = int(input("Mean： "))
var = int(input("Variance： "))

print(f'Data point source function: N({mean}, {var})\n')

flag = True

data = []
while flag:
    val = UnivariateGaussian(mean, var)
    print(f'Add data point: {val}')
    data.append(val)
    new_mean = np.mean(data)
    new_var = np.var(data)
    print(f'Mean = {new_mean}   Variance = {new_var}')
    
    if np.abs(new_mean - mean) < 0.1 and np.abs(new_var - var) < 0.1:
        flag = False