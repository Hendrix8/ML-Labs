import numpy as np 
from numpy import linalg as la 
import random 

n = 2
epsilon = 10 ** (-20)

A = np.array([[1,-3],
              [-3,1]])

def approx(A, n):

    # random vector 
    z = np.array([random.randint(1,100) for i in range(n)])
    z = z/la.norm(z, np.inf) # noramlize with || ||oo
    
    y = z 
    yn = np.array([random.uniform(1,100) for i in range(n)])

    while la.norm((yn - y), np.inf) >= epsilon:
        x = la.solve(A, y)
        y = yn
        yn = x/la.norm(x, np.inf)

    Ay = np.dot(A,y)

    return Ay/la.norm(Ay) #la.norm(np.dot(A,yn),np.inf), np.dot(A,yn)/la.norm(np.dot(A,y),2)

print(approx(A,n))

