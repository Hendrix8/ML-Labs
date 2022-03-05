# Christos Panourgias
# AM: 2405

# This program contains tools that I will need for the exercises 

import numpy as np 
from numpy import linalg as la 
from scipy import stats as st

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

        # taking the values of numbInList in pairs but not including the pairs between the spaces
        L = [[numbInList[i], numbInList[i + 1]] for i in range(len(numbInList) - 1) if i%2 == 0]
        
        return L

    # gets a result from nx.pagerank and
    # returns a pagerank that has 
    # {1: 4, 2: 3, ...} like dict where left is the rank and to the right is the node
    def toNumberRanks(self, pgRank):
        rankList = list(pgRank.values())
        
        rawRank = len(rankList) - st.rankdata(rankList) + 1 
        rawRank = [int(i) for i in rawRank]
        indexes = [i + 1 for i in range(len(rawRank))]
        
        dictRank = dict(zip(rawRank, indexes))
        dictRank = dict(sorted(dictRank.items())) # sorting the dictionary so that the ranks are in ascending order

        return dictRank

    def graphAnalyse(self, g):

        # initializing some values
        d = 0.85
        epsilon = 10**(-6)
        kmax = 10000

        # creating the L(j) function
        def Lj(L, j):
            cnt = 0
            for tupl in L:
                if j == tupl[0]:
                    cnt += 1
                else:
                    pass 
            return cnt


        L = self.txtToGraph(g)  # [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (4, 1), (4, 3)]
        # L also represents the positions in which the items 1/L(j) go in the form (column, row)

        siteNumber = len(dict(L)) # the number of websites are the length of first items of the L dictionary. (dict deletes duplicates)
        A = np.zeros((siteNumber,siteNumber)) # creating an empty A to later add the elements

        # creating matrix A 
        for tupl in L:
            A[tupl[1] - 1][tupl[0] - 1] = 1 / Lj(L, tupl[0]) # adding in the form (column,row) as L represents

        # creating the matrix M 
        M = d * A + ( (1 - d) / siteNumber ) * np.ones((siteNumber, siteNumber))
        
        lamda, x = self.powerMethod(M, kmax, epsilon)[0:2]

        return lamda, x