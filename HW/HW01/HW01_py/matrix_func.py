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

def GetCofactor(input_m, p, q):
    n = len(input_m)

    tmp1 = []
    tmp2 = []

    for i in range(0, n):
        for j in range(0, n):
            if i != p and j != q:
                tmp1.append(input_m[i][j])

    for i in range(0, len(tmp1), n-1):
        tmp2.append(tmp1[i:i+n-1])

    return tmp2

def Determinant(input_m):
    n = len(input_m)
    
    if n == 1:
        return input_m[0][0]

    sign = 1
    
    val = 0

    for i in range(0, n):
        tmp = GetCofactor(input_m, 0, i)
        val += sign * input_m[0][i] * Determinant(tmp)
        
        sign = -sign
        
    return val

def GetAdjoint(input_m):
    n = len(input_m)
    tmp = []
    
    for i in range(0, n):
        tmp.append([])
        for j in range(0, n):
            cof = GetCofactor(input_m, i, j)
            val = pow(-1, i+j) * Determinant(cof)
            tmp[i].append(val)
    tmp = Transpose(tmp)
    
    return tmp

# 反矩陣
def Inverse(input_m):
    determinant = Determinant(input_m)
    if determinant == 0:
        print("It's a singular matrix")
        return
    
    adjoint_m = GetAdjoint(input_m)
    inverse_m = MultipleMatrix(adjoint_m, 1/determinant)
    
    return inverse_m

# 三角矩陣行列式值
def TriangleDeterminant(T):
    n = len(T)
    det = 1
    
    # 對角相乘
    for i in range(0, n):
        det *= T[i][i]
    return det

def InverseLower(L):
    tmp = []
    n = len(L)
    det = TriangleDeterminant(L)
    if det == 0:
        print("It's a singular matrix")
        return
    
    for i in range(0, n):
        tmp.append([])
        for j in range(0, n):
            val = 0
            if i > j:
                cof = GetCofactor(L, j, i)
                val = pow(-1, i+j) * Determinant(cof) / det
            elif i == j:
                val = 1 / L[i][j]
            tmp[i].append(val)
    return tmp

def InverseUpper(U):
    tmp = []
    n = len(U)
    det = TriangleDeterminant(U)
    if det == 0:
        print("It's a singular matrix")
        return
    
    for i in range(0, n):
        tmp.append([])
        for j in range(0, n):
            val = 0
            if i < j:
                cof = GetCofactor(U, j, i)
                val = pow(-1, i+j) * Determinant(cof) / det
            elif i == j:
                val = 1 / U[i][j]
            tmp[i].append(val)
    return tmp