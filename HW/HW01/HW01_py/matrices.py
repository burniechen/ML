from matrix_func import *

# Design Matrix
def GetDesignMatrix(input_m, n):
    design_m = []
    tmp_m = []

    for ele in input_m:
        for p in range(0, n):
            # 升次
            tmp_m.append(pow(ele[0], p))
        design_m.append(tmp_m)
        tmp_m = []
    
    return design_m

# Regular Matrix
def GetRegularMatrix(input_m, lam):
    input_m_t = Transpose(input_m)
    multify_m = MultifyMatrix(input_m_t, input_m)
    length = len(multify_m)
    for i in range(0, length):
        multify_m[i][i] += lam
    return multify_m