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
    hession_m = np.linalg.inv(MultipleMatrix(AtA, 2)).tolist()
    
    return hession_m

def NewtonMethod(data, power):
    x_bef = np.random.uniform(low=min(data)[0], high=max(data)[1], size=(power,1)).tolist()
    
    error = 99.9
    
    while(error > 0.01):
        gradient_m = GradientMatrix(data, power, x_bef)
        hession_m = HessionMatrix(data, power)

        x_aft = SubtractMatrix(x_bef, MultifyMatrix(hession_m, gradient_m))

        error = LSEerrorMatrix(x_aft, x_bef)
        
        x_bef = x_aft
    
    return x_aft