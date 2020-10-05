# 矩陣相加
def AddMatrix(a, b):
    tmp = []
    x = len(a)
    y = len(a[0])
    
    for i in range(0, x):
        tmp.append([])
        for j in range(0, y):
            val = a[i][j] + b[i][j]
            tmp[i].append(val)
    return tmp

# 矩陣相減
def SubtractMatrix(a, b):
    tmp = []
    x = len(a)
    y = len(a[0])
    
    for i in range(0, x):
        tmp.append([])
        for j in range(0, y):
            val = a[i][j] - b[i][j]
            tmp[i].append(val)
    return tmp

# 矩陣乘倍數
def MultipleMatrix(input_m, k):
    x = len(input_m)
    y = len(input_m[0])
    tmp = []
    
    for i in range(0, x):
        tmp.append([])
        for j in range(0, y):
            val = k * input_m[i][j]
            tmp[i].append(val)
    return tmp

# 矩陣相乘
def MultifyMatrix(a, b):
    tmp = []
    x = len(a)
    y = len(b[0])
    n = len(b)

    for i in range(0, x):
        tmp.append([])
        for j in range(0, y):
            val = 0
            for k in range(0, n):
                val += a[i][k] * b[k][j]
            tmp[i].append(val)
    
    return tmp
    
# 矩陣轉置
def Transpose(input_m):
    x = len(input_m)
    y = len(input_m[0])

    tmp = []
    for i in range(0, y):
        tmp.append([])
        for j in range(0, x):
            tmp[i].append(input_m[j][i])

    return tmp

# ＬＵ分解，回傳（上三角，下三角）
def LUdecomp(input_m):
    lower_m = []

    length = len(input_m)
    
    # 初始lower_m = 單位矩陣
    for i in range(0, length):
        lower_m.append([])
        for j in range(0, length):
            if i == j:
                lower_m[i].append(1)
            else:
                lower_m[i].append(0)

    # 次數
    for k in range(0, length - 1):
        # 高斯運算
        for i in range(k+1, length):
            # 找倍數
            multiple = input_m[i][k] / input_m[k][k]
            lower_m[i][k] = multiple

            for j in range(k, length):
                input_m[i][j] = input_m[i][j] - input_m[k][j] * multiple
    
    
    return [lower_m, input_m]

# 利用ＬＵ分解求解
def GetSolByLU(regular_m, b):
    A = LUdecomp(regular_m)
    L = A[0]
    U = A[1]

    # Ly = b，先求y，上到下
    y = []
    for i in range(0, len(L)):
        val = b[i][0]
        for j in range(0, i):
            val -= L[i][j] * y[j][0]
        y.append([val])
    
    # Ux = y，在求x，下到上
    x = []
    for i in range(0, len(U)):
        x.append([0])

    for i in range(len(U), 0, -1):
        val = y[i-1][0]
        for j in range(len(U), i, -1):
            val -= U[i-1][j-1] * x[j-1][0]
        x[i-1][0] = val / U[i-1][i-1]
    
    return x