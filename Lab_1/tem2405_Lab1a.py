# Christos Panourgias 
# AM: 2405

import numpy as np 
from numpy import linalg as la, power, sort 
from methods import Method

epsilon = 10**(-6) # error 
kmax = 10000 # maximum iterations 

pM = Method(epsilon, kmax) # power method instance

# EXAMPLE 
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])


lamda,x, dk_new, dk_old= pM.powerMethod(A)

# comparing power method results with original results
print("------------------------------------------------------------")
print("                       MATRIX A                             ")
print("------------------------------------------------------------\n")

print("Largest Eigen Value λ = ",max(la.eig(A)[0]))
print("Largest Eigen Value approximation ( power method ) λ = ", lamda)

# finding λ2 
eigSorted = sort(la.eig(A)[0].tolist()) # sorting the list of eigenvalues to easily find lamda2
lamda_2 = eigSorted[-2]
lamda_1 = eigSorted[-1]

print("\n|λ2 / λ1| = ", abs(lamda_2 / lamda_1))
print("d_k+1 / dk = ", dk_new / dk_old, "\n")

print("------------------------------------------------------------")
print("                       MATRIX T                             ")
print("------------------------------------------------------------\n")
# Creating matrix T
n = np.random.randint(1,100) # choosing a random dimemnsion from 1 to 99 
T = np.diag(2*np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
lamda_T = pM.powerMethod(T)[0]
print("Largest Eigen Value ( linalg ) λ = ",max(la.eig(T)[0]))
print("Largest Eigen Value ( using the given formula -> max{ 2 - 2cos(kπ/(n+1)) | k = 1,...,n } ) λ = ",max([2 - 2*np.cos( (k * np.pi) / (len(T) + 1) ) for k in range(n)]))
print("Largest Eigen Value approximation ( power method ) λ = ", lamda_T, "\n")
print("------------------------------------------------------------")

