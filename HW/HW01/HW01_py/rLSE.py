from matrix_func import *
from matrices import *

def rLSE(data, power, lam):
    design_m = GetDesignMatrix(data, power)
    design_m_t = Transpose(design_m)

    b = []
    for ele in data:
        b.append([ele[1]])
    
    # AtA + λI
    regular_m = GetRegularMatrix(design_m, lam)
    
    # Atb
    design_m_txb = MultifyMatrix(design_m_t, b)
    
    L, U = LUdecomp(regular_m)

    inv_L = InverseLower(L)
    inv_U = InverseUpper(U)
    
    inv_regular_m = MultifyMatrix(inv_U, inv_L)
    x = MultifyMatrix(inv_regular_m, design_m_txb)
    return x