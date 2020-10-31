import numpy as np

def UnivariateGaussian(mean, var):
    flag = True
    while flag:
        U = np.random.uniform(-1, 1)
        V = np.random.uniform(-1, 1)
        S = U**2 + V**2
        
        flag = False
        if S >= 1:
            flag = True

    coef = (-2 * np.log(S)) / S
    X = U * np.sqrt(coef)
    # Y = V * np.sqrt(coef)
    
    return np.sqrt(var) * X + mean

def PolynomialBasis(n, w, a):
    x = np.random.uniform(-1, 1)
    tmp = 0
    for i in range(n):
        tmp += w[i] * pow(x, i)
    e = UnivariateGaussian(0, a)
    y = tmp + e
    
    return x, y