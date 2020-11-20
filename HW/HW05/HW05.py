import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy.optimize import minimize

def RQkernel(xa, xb, var=1, alpha=1, lengthscale=1):
    return var * (1 + ((xa - xb) ** 2) / (2 * alpha * (lengthscale ** 2))) ** (-alpha)

def GetLogLikelihood(args):
    var, alpha, lengthscale = args
    K = np.zeros((row, row))
    for i in range(row):
        for j in range(row):
            K[i][j] = RQkernel(X[i], X[j], var, alpha, lengthscale)
            
    return -(-0.5 * Y.T @ np.linalg.inv(K) @ Y -0.5 * np.log(np.abs(np.linalg.det(K))) - (n/2) * np.log(2 * np.pi)) 

f = open("input.data")
input_data = f.read()
f.close()

row = 34
col = 2
n = 34
data = np.zeros((row, col))
count = 0
for xy in input_data.split("\n"):
    if xy != '':
        tmpX, tmpY = xy.split(" ")
        data[count] = [float(tmpX), float(tmpY)]
    count += 1

X = data[:, 0]
Y = data[:, 1]

K = np.zeros((row, row))
var = 2.5
a = 500
l = 1

for i in range(row):
    for j in range(row):
        K[i][j] = RQkernel(X[i], X[j], var, a, l)

sample = np.linspace(-60, 60, 1000)
mean_y = []
var_y = []
Ks = np.zeros((row))
b = 5
for xs in sample:
    for i in range(row):
        Ks[i] = RQkernel(xs, X[i], var, a, l)

    Kss = RQkernel(xs, xs, var, a, l) + 1/b
    mean_s = Ks @ np.linalg.inv(K) @ Y
    var_s = Kss - Ks @ np.linalg.inv(K) @ Ks.T

    mean_y.append(mean_s)
    var_y.append(var_s)

mean_y = np.array(mean_y)
var_y = np.array(var_y)

# Plotting
x_m = np.arange(1, 101, 1)
L_var_m = []
L_alpha_m = []
L_length_m = []

for i in x_m:
    L = GetLogLikelihood([i, 1, 1])
    L_var_m.append(L)
    L = GetLogLikelihood([1, i, 1])
    L_alpha_m.append(L)
    L = GetLogLikelihood([1, 1, i])
    L_length_m.append(L)

plt.figure(figsize=(10, 8))

plt.subplot(221)
plt.title("Gaussian Process Regression")
plt.plot(X, Y, '.')
plt.plot(sample, mean_y, 'r')
plt.fill_between(sample, mean_y - 1.96*(var_y**0.5), mean_y + 1.96*(var_y**0.5), color='pink')

plt.subplot(222)
plt.title("Parameter: variance")
plt.plot(x_m, L_var_m)

plt.subplot(223)
plt.title("Parameter: alpha")
plt.plot(x_m, L_alpha_m)

plt.subplot(224)
plt.title("Parameter: lengthscale")
plt.plot(x_m, L_length_m)

plt.show()

# Minimize negative marginal log-likelihood
begin = np.array([1, 1, 1])
res = minimize(fun=GetLogLikelihood, x0=begin, method='SLSQP')
print(f'Negative marginal log-likelihood：{res.fun}')
print(f'Variance：{res.x[0]}')
print(f'Alpha：{res.x[1]}')
print(f'Lengthscale：{res.x[2]}')