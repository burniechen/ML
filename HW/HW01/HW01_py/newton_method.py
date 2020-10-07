import numpy as np

from matrix_func import *
from matrices import *
from LSEerror_func import LSEerrorMatrix

# Gradient Matrix
def GradientMatrix(data, power, x_bef):
    design_m = GetDesignMatrix(data, power)
    design_m_t = Transpose(design_m)
    
    b = []
    for ele in data:
        b.append([ele[1]])
    
    AtA = MultifyMatrix(design_m_t, design_m)
    AtAx = MultifyMatrix(AtA, x_bef)
    
    Atb = MultifyMatrix(design_m_t, b)
    
    gradient_m = MultipleMatrix(SubtractMatrix(AtAx, Atb) ,2)
    
    return gradient_m

# Hession Matrix
def HessionMatrix(data, power):
    design_m = GetDesignMatrix(data, power)
    design_m_t = Transpose(design_m)
    AtA = MultifyMatrix(design_m_t, design_m)
    
    L, U = LUdecomp(MultipleMatrix(AtA, 2))

    inv_L = InverseLower(L)
    inv_U = InverseUpper(U)
    
    hession_m = MultifyMatrix(inv_U, inv_L)
    
    return hession_m

def NewtonMethod(data, power):
    x_bef = np.random.uniform(low=min(data)[0], high=max(data)[1], size=(power,1)).tolist()
    
    gradient_m = GradientMatrix(data, power, x_bef)
    hession_m = HessionMatrix(data, power)

    x_aft = SubtractMatrix(x_bef, MultifyMatrix(hession_m, gradient_m))
    
    return x_aft