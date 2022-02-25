import numpy as np 

# Askisi 1

A = np.array([ [1, 2, 3], 
               [4, 5, 6],
               [7, 8, 9]  ])

def sum_elements(A):
    elements = A.flatten()
    return sum(elements)

print("The sum of elements of A is: %d" %sum_elements(A))

## 2os tropos 
def sumElements(A):
    return(sum(map(sum,A)))

print("The sum of elements of A is: %d" %sumElements(A))


# ----------

# Askhsh 2
# Shmeiwsh: A[rows,columns]

def matrix_extend(A):
    newRow = [] 
    newColumn = []

    # creating the new column and row
    for i in A:
        newColumn.append(sum(i))
    
    for i in range(len(A)):
        column = A[:,i]
        newRow.append(sum(column))

    # adding the diagonal sum to the new row
    newRow.append(sum(np.diag(A)))

    # adding extra column
    A = np.insert(A, len(A) , newColumn, axis = 1)

    # adding extra row
    A = np.insert(A, len(A), newRow, axis = 0)
    
    return A

print(matrix_extend(A))


