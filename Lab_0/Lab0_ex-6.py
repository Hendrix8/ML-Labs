import numpy as np 
from numpy import linalg as la
import random 

n = int(input("Give n: "))

A = np.array([random.randint(1,10) for i in range(n**2)])
A = A.reshape(n,n)

print(A)
print(la.norm(A, 1))
print(la.norm(A, 2))
print(la.norm(A, np.inf))
print(la.inv(A))