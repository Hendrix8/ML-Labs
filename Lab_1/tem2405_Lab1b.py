# Christos Panourgias
# AM: 2405

import numpy as np 
from numpy import linalg as la 
from methods import Method

def graphAnalyse(g):

    # initializing some values
    d = 0.85
    epsilon = 10**(-6)
    kmax = 10000
    pM = Method(epsilon, kmax)

    # creating the L(j) function
    def Lj(L, j):
        cnt = 0
        for tupl in L:
            if j == tupl[0]:
                cnt += 1
            else:
                pass 
        return cnt

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

    # L also represents the positions in which the items 1/L(j) go in the form (column, row)

    siteNumber = len(dict(L)) # the number of websites are the length of first items of the L dictionary. (dict deletes duplicates)
    A = np.zeros((siteNumber,siteNumber)) # creating an empty A to later add the elements

    # creating matrix A 
    for tupl in L:
        A[tupl[1] - 1][tupl[0] - 1] = 1 / Lj(L, tupl[0]) # adding in the form (column,row) as L represents

    # creating the matrix M 
    M = d * A + ( (1 - d) / siteNumber ) * np.ones((siteNumber, siteNumber))
    
    lamda, x = pM.powerMethod(M)[0:2]
    return lamda, x

g1 = open("graph0.txt", "r") 
g2 = open("graph1.txt", "r")
g3 = open("graph2.txt", "r")

g1_lamda, g1_vector = graphAnalyse(g1)
g2_lamda, g2_vector = graphAnalyse(g2)
g3_lamda, g3_vector = graphAnalyse(g3)

print("\n---------------------------------------------------------------------------------------------")
print("                                     GRAPH 1                                                 ")
print("---------------------------------------------------------------------------------------------")

print("The eigenvalue and eigenvector of M of the first graph approximated with power method are:\n"+
"Eigenvalue = ", g1_lamda, "\n" +
"Eigenvector = ", g1_vector,"\n")
print("The sum of elements of the eigenvector is = ", sum(g1_vector))

print("---------------------------------------------------------------------------------------------")
print("                                     GRAPH 2                                                 ")
print("---------------------------------------------------------------------------------------------")

print("The eigenvalue and eigenvector of M of the second graph approximated with power method are:\n"+
"Eigenvalue = ", g2_lamda, "\n" +
"Eigenvector = ", g2_vector, "\n")
print("The sum of elements of the eigenvector is = ", sum(g2_vector))


print("---------------------------------------------------------------------------------------------")
print("                                     GRAPH 3                                                 ")
print("---------------------------------------------------------------------------------------------")

print("The eigenvalue and eigenvector of M of the third graph approximated with power method are:\n"+
"Eigenvalue = ", g3_lamda, "\n" +
"Eigenvector = ", g3_vector, "\n")
print("The sum of elements of the eigenvector is = ", sum(g3_vector))

print("---------------------------------------------------------------------------------------------")
print("                                     THE END                                                 ")
print("---------------------------------------------------------------------------------------------\n")

