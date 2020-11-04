import numpy as np
import matplotlib.pyplot as plt

from data_generator import UnivariateGaussian, PolynomialBasis

choose_type = int(input("0 for Univariate Gaussian, 1 for Polynomail Basis Linear Model： "))

if choose_type == 0:
    mean = float(input("Mean： "))
    var = float(input("Variance： "))

    points = []
    for i in range(10000):
        points.append(UnivariateGaussian(mean, var))
    
elif choose_type == 1:
    n = int(input("basis: "))
    w = []
    for i in range(n):
        w.append(float(input("Coefficient of x^" + str(i) + "： ")))
    a = float(input("Variance： "))

    points = []
    for i in range(10000):
        points.append(PolynomialBasis(n, w, a)[1])

plt.figure(figsize=(10, 8))
plt.hist(points, 100)
plt.show()