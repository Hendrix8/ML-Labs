import numpy as np 
from numpy import linalg as la 
from tem2405_Lab1a import Method

g = open("graph0.txt", "r") 

# Extracting the information of the file 
lines = g.readlines() # example ['1 2\n', '1 3\n', '1 4\n', '2 3\n', '2 4\n', '3 1\n', '4 1\n', '4 3\n']
L = []
for i in lines:
    L.append( (int(i[0]), int(i[2])) ) # getting only 0th and 2nd element from lines 
    # L also represents the positions in which the items 1/L(j) go in the form (column, row)

dimA = len(dict(L)) # the dimensions of A are the card of first items of the L list. (dict deletes duplicates)


# creating the L(j) function
def Lj(L, j):
    cnt = 0
    for tupl in L:
        if j == tupl[0]:
            cnt += 1
        else:
            pass 
    return cnt

A = np.zeros((dimA,dimA)) # creating an empty A to later add the elements

# creating matrix A 
for tupl in L:
    A[tupl[1] - 1][tupl[0] - 1] = 1 / Lj(L, tupl[0]) # adding in the form (column,row) as L represents

print(A)
    

