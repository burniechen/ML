import numpy as np
import matplotlib.pyplot as plt

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
    Y = V * np.sqrt(coef)
    
    return [np.sqrt(var) * X + mean, np.sqrt(var) * Y + mean]

def LogisticF(w, A):
    return 1 / (1 + np.exp(- (A @ w)))

def GradientDescent(w, A, Y, r=0.001):
    flag = True
    while(flag):
        w_new = w + r * A.T @ (Y - LogisticF(w, A))

        if np.linalg.norm(A.T @ (Y - LogisticF(w, A))) < 0.01:
            flag = False
        else:
            w = w_new
    
    return w

def GetHessian(w, A):
    row, col = A.shape

    D = np.zeros((row, row))
    for i in range(row):
        val = np.exp(-(A[i] @ w))
        D[i][i] = val / ((1 + val) ** 2)
        
    Hessian = A.T @ D @ A
    rank = np.linalg.matrix_rank(Hessian)
    
    return Hessian, rank

def NewtonMethod(w, A, Y, r=0.001):
    flag = True
    while(flag):
        Hessian, rank = GetHessian(w, A)
        if rank == 3:
            Hessian_inv = np.linalg.inv(Hessian)
            w_new = w + r * (Hessian_inv @ A.T @ (Y - LogisticF(w, A)))

            if np.linalg.norm(Hessian_inv @ A.T @ (Y - LogisticF(w, A))) < 0.01:
                flag = False
            else:
                w = w_new
    else:
        w = GradientDescent(w, A, Y, r)

    return w

def Plot(D1, D2, w_in, A, title):
    plt.title(title)

    if np.array_equal(w_in, w_init):
        plt.plot(D1[:, 0], D1[:, 1], 'o', color='red')
        plt.plot(D2[:, 0], D2[:, 1], 'o', color='blue')
    else:
        sample = np.array([D1, D2]).reshape(n*2, 2)
        y_pred = LogisticF(w_in, A).reshape(n*2, )
        cluster1 = sample[(y_pred < 0.5)]
        cluster2 = sample[(y_pred >= 0.5)]

        predict_c1 = cluster1[:, 0].size
        predict_c2 = cluster2[:, 0].size

        print(f'\n{title}:\n')
        print("w:")
        for ele in w_in:
            print(ele[0])

        print("\nConfusion Matrix:")
        print("             Predict cluster 1 Prdict cluster 2")
        print(f'Is cluster 1       {predict_c1}                  {n - predict_c1}')
        print(f'Is cluster 2       {n - predict_c2}                  {predict_c2}\n')

        print(f'Sensitivity (Successfully predict cluster 1): {predict_c1 / n}')
        print(f'Specificity (Successfully predict cluster 2): {predict_c2 / n}')

        plt.plot(cluster1[:, 0], cluster1[:, 1], 'o', color='red')
        plt.plot(cluster2[:, 0], cluster2[:, 1], 'o', color='blue')

n = int(input("Number of samples: "))

D1_m = float(input("Data 1 - Mean: "))
D1_v = float(input("Data 1 - Variance: "))

D2_m = float(input("Data 2 - Mean: "))
D2_v = float(input("Data 2 - Variance: "))

D1 = np.zeros((n, 2))
D2 = np.zeros((n, 2))

w_init = np.zeros((3, 1))
w = w_init

A = np.zeros((n*2, 3))
for i in range(n):
    D1[i] = UnivariateGaussian(D1_m, D1_v)
    A[i] = [1, D1[i][0], D1[i][1]]
    
    D2[i] = UnivariateGaussian(D2_m, D2_v)
    A[i+n] = [1, D2[i][0], D2[i][1]]

Y = np.array([np.zeros((n)), np.zeros((n)) + 1]).reshape(n*2, 1)

# Plot region
plt.figure(figsize=(10, 8))

plt.subplot(131)
w = w_init
Plot(D1, D2, w, A, "Ground Truth")

plt.subplot(132)
w = w_init
w = GradientDescent(w, A, Y, 0.001)
Plot(D1, D2, w, A, "Gradient Descent")

plt.subplot(133)
w = w_init
w = NewtonMethod(w, A, Y, 0.001)
Plot(D1, D2, w, A, "Newton's Method")

plt.show()