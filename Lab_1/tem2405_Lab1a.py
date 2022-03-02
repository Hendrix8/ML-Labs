# Christos Panourgias 
# AM: 2405

import numpy as np 
from numpy import linalg as la, power, sort 

def powerMethod(A):

    x = np.array([np.random.uniform(0,10) for i in range(len(A))]) # creating a random vector of dimensions n and elements in range [0,9)
    x = x / la.norm(x, 1)  # normalizing x using the 1-norm

    kmax = 10000 # maximum number of iterations
    epsilon = 10**(-6) # error 

    # initializing k and dk 
    k = 0 
    dk = 1

    while dk > epsilon and k < kmax:
    
        x_old = x # saving x to calculate dk

        x = A @ x # resetting x
        x = x / la.norm(x,1)

        dk_old = dk # saving dk for later use
        dk = la.norm(x - x_old, 1)
    
    lamda = (x.T @ A @ x) / (x.T @ x)

    return lamda, dk, dk_old, k

# EXAMPLE 
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

# comparing power method results with original results
print("------------------------------------------------------------")
print("\nMATRIX A: \n")
print("Largest Eigen Value λ = ",max(la.eig(A)[0]))
print("Largest Eigen Value approximation ( power method ) λ = ", powerMethod(A)[0])

# finding λ2 
eigSorted = sort(la.eig(A)[0].tolist()) # sorting the list of eigenvalues to easily find lamda2
lamda_2 = eigSorted[-2]
lamda_1 = eigSorted[-1]

## TODO: tsekare an einai lathos to dk+1/dk kai tha eprepe na einai (d_k+1/dk)**k
print("\n|λ2 / λ1| = ", abs(lamda_2 / lamda_1))
print("d_k+1 / dk = ", (powerMethod(A)[2] / powerMethod(A)[1]), "\n")

print("MATRIX T: \n")

# Creating matrix T
n = np.random.randint(1,100) 
T = np.diag(2*np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)

print("Largest Eigen Value ( linalg ) λ = ",max(la.eig(T)[0]))
print("Largest Eigen Value ( using the given formula -> max{ 2 - 2cos(kπ/(n+1)) | k = 1,...,n } ) λ = ",max([2 - 2*np.cos( (k * np.pi) / (len(T) + 1) ) for k in range(n)]))
print("Largest Eigen Value approximation ( power method ) λ = ", powerMethod(T)[0], "\n")
print("------------------------------------------------------------")

