import numpy as np 
from numpy import linalg as la
import random

# Askhsh 7 
def func(n):
    A = np.array([random.randint(0,10) for i in range(n)])
    B = np.array([random.randint(0,10) for i in range(n)])

    # testing matices
    '''A=np.array([[1,1],[1,2]])
    B=np.array([[1,0],[0,2]])'''
    
    b = np.ones(n)

    # testing array 
    '''b = np.ones(2)'''

    return la.solve(A,b)

