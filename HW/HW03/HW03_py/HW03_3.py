import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from data_generator import PolynomialBasis

def GetDesignMatrix(x, n):
    x_m = []
    for i in range(n):
        x_m.append(pow(x, i))
    return np.array(x_m).reshape(1, n)

def PlotIncomes(incomes, incomes_m, incomes_v, num):
    plt.xlim(-2, 2)
    plt.ylim(-30, 30)

    x_points = np.linspace(-2, 2, num)

    mean_y = []
    var_y = []
    for p in x_points:
        P = GetDesignMatrix(p, n)
        val = P.dot(incomes_m)
        mean_y.append(val)

        Pt = np.transpose(P)
        val = a_var + P.dot(incomes_v).dot(Pt)
        var_y.append(val)

    mean_y = np.array(mean_y).reshape(num, )
    var_y = np.array(var_y).reshape(num, )
        
    plt.plot(incomes[:, 0], incomes[:, 1], '.')
    plt.plot(x_points, mean_y, 'k')
    plt.plot(x_points, mean_y + var_y, 'r')
    plt.plot(x_points, mean_y - var_y, 'r')

b = int(input("b: "))
n = int(input("n: "))
a_var = float(input("a: "))
w = np.zeros((n, )) # 1*n
for i in range(n):
    w[i] = float(input("w" + str(i) + ": "))

a = 1/a_var
S = b * np.identity(n)

incomes_10 = np.zeros((10, 2))
incomes_50 = np.zeros((50, 2))
all_incomes = []

prior_mean = np.zeros((n, 1))
prior_var = inv(S)

flag = True
iteration = 0

while flag:
    x, y = PolynomialBasis(n, w, a_var)
    X = GetDesignMatrix(x, n)
    Xt = X.reshape(n, 1)

    lam = a * Xt.dot(X) + S

    posterior_mean = inv(lam).dot((a * Xt * y + S.dot(prior_mean)))
    posterior_var = inv(lam)
    print(posterior_mean)

    predict_mean = X.dot(prior_mean)
    predict_var = a_var + X.dot(inv(lam)).dot(Xt)
    print(f'({predict_mean[0][0]}, {predict_var[0][0]})')

    prior_mean = posterior_mean
    prior_var = posterior_var
    S = inv(posterior_var)

    all_incomes.append([x, y])

    iteration += 1

    if iteration < 11:
        incomes_10[iteration - 1] = [x, y]
        incomes_10_m = posterior_mean
        incomes_10_v = posterior_var

    if iteration < 51:
        incomes_50[iteration - 1] = [x, y]
        incomes_50_m = posterior_mean
        incomes_50_v = posterior_var
    
    if iteration > 50 and np.abs(predict_var[0][0] - a_var) < 0.005:
        all_incomes = np.array(all_incomes)
        flag = False
        print(iteration)

num = 100
plt.figure(figsize=(10, 8))

ax1 = plt.subplot(221)
ax1.set_title("Ground Truth")
PlotIncomes(np.array([[[], []]]), w, 0, num)

ax2 = plt.subplot(222)
ax2.set_title("Predict Result")
PlotIncomes(all_incomes, posterior_mean, posterior_var, num)

ax3 = plt.subplot(223)
ax3.set_title("After 10 incomes")
PlotIncomes(incomes_10, incomes_10_m, incomes_10_v, num)

ax4 = plt.subplot(224)
ax4.set_title("After 50 incomes")
PlotIncomes(incomes_50, incomes_50_m, incomes_50_v, num)

plt.show()