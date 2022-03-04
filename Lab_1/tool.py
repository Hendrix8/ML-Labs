# Christos Panourgias
# AM: 2405

import numpy as np 
from numpy import linalg as la 

class Tool:

    def __init__(self):
        pass


    def powerMethod(self, A, kmax, epsilon):

        x = np.array([np.random.uniform(1,10) for i in range(len(A))]) # creating a random vector of dimensions n and elements in range [1,9)
        x = x / la.norm(x, 1)  # normalizing x using the 1-norm

        # initializing k and dk 
        k = 0 
        dk = 1

        while dk > epsilon and k < kmax:
        
            x_old = x # saving x to calculate dk

            x = A @ x # resetting x
            x = x / la.norm(x,1)

            dk_old = dk # saving dk for later use
            dk = la.norm(x - x_old, 1)

            k += 1
        
        lamda = (x.T @ A @ x) / (x.T @ x)

        return lamda, x, dk, dk_old

    def txtToGraph(self, g):

        # Extracting the information of the file 
        lines = g.readlines() # example ['1 2\n', '1 3\n', '1 4\n', '2 3\n', '2 4\n', '3 1\n', '4 1\n', '4 3\n']

        numbInList = []

        for i in lines:
            for j in i.split(): # ['1', '2']... 
                if j.isdigit():
                    numbInList.append(int(j)) # putting all the numbers in the string 

        x = []
        y = []
        for i in range(len(numbInList)): 
            # grouping them in pairs and adding them in the L list
            if i%2 == 0:
                x.append(numbInList[i]) # [1, 1, 1, 2, 2, 3, 4, 4]
            else:
                y.append(numbInList[i]) # [2, 3, 4, 3, 4, 1, 1, 3]

        
        L = list(zip(x, y)) # [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (4, 1), (4, 3)]

        return L