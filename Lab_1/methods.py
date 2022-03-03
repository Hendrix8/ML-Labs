# Christos Panourgias
# AM: 2405

import numpy as np 
from numpy import linalg as la 

class Method:

    def __init__(self, epsilon, kmax):
        self.epsilon = epsilon
        self.kmax = kmax


    def powerMethod(self, A):

        x = np.array([np.random.uniform(1,10) for i in range(len(A))]) # creating a random vector of dimensions n and elements in range [1,9)
        x = x / la.norm(x, 1)  # normalizing x using the 1-norm

        # initializing k and dk 
        k = 0 
        dk = 1

        while dk > self.epsilon and k < self.kmax:
        
            x_old = x # saving x to calculate dk

            x = A @ x # resetting x
            x = x / la.norm(x,1)

            dk_old = dk # saving dk for later use
            dk = la.norm(x - x_old, 1)

            k += 1
        
        lamda = (x.T @ A @ x) / (x.T @ x)

        return lamda, x, dk, dk_old