# Christos Panourgias 
# AM: 2405

import numpy as np 
from numpy import linalg as la, power, sort 
import methods

epsilon = 10**(-6) # error 
kmax = 10000 # maximum iterations 

pM = methods.Method(epsilon, kmax) # power method instance

# EXAMPLE 
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

# comparing power method results with original results
print("------------------------------------------------------------")
print("\nMATRIX A: \n")
print("Largest Eigen Value λ = ",max(la.eig(A)[0]))
print("Largest Eigen Value approximation ( power method ) λ = ", pM.powerMethod(A)[0])

# finding λ2 
eigSorted = sort(la.eig(A)[0].tolist()) # sorting the list of eigenvalues to easily find lamda2
lamda_2 = eigSorted[-2]
lamda_1 = eigSorted[-1]

## TODO: tsekare an einai lathos to dk+1/dk kai tha eprepe na einai (d_k+1/dk)**k
print("\n|λ2 / λ1| = ", abs(lamda_2 / lamda_1))
print("d_k+1 / dk = ", (pM.powerMethod(A)[2] / pM.powerMethod(A)[1]), "\n")

print("MATRIX T: \n")

# Creating matrix T
n = np.random.randint(1,100) # choosing a random dimemnsion from 1 to 99 
T = np.diag(2*np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)

print("Largest Eigen Value ( linalg ) λ = ",max(la.eig(T)[0]))
print("Largest Eigen Value ( using the given formula -> max{ 2 - 2cos(kπ/(n+1)) | k = 1,...,n } ) λ = ",max([2 - 2*np.cos( (k * np.pi) / (len(T) + 1) ) for k in range(n)]))
print("Largest Eigen Value approximation ( power method ) λ = ", pM.powerMethod(T)[0], "\n")
print("------------------------------------------------------------")

