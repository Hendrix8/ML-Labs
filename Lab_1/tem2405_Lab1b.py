import numpy as np 
from numpy import linalg as la 

g = open("graph0.txt", "r") 

# Extracting the information of the file 
lines = g.readlines() #['1 2\n', '1 3\n', '1 4\n', '2 3\n', '2 4\n', '3 1\n', '4 1\n', '4 3\n']
L = []
for i in lines:
    L.append( ( int(i[0]), int(i[2])) ) # getting only 0th and 2nd element from lines 

dimA = len(dict(L)) # the dimensions of A are the card of first items of the L list. (dict deletes duplicates)

A = np.zeros(dimA,dimA)
# creating matrix A 
print(A)
    

