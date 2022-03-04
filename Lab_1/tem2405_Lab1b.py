# Christos Panourgias
# AM: 2405

import numpy as np 
from numpy import linalg as la 
from tool import Tool
from scipy import stats as st

tool = Tool()
g1 = open("graph0.txt", "r") 
g2 = open("graph1.txt", "r")
g3 = open("graph2.txt", "r")

g1_lamda, g1_vector = tool.graphAnalyse(g1)
g2_lamda, g2_vector = tool.graphAnalyse(g2)
g3_lamda, g3_vector = tool.graphAnalyse(g3)

print("\n---------------------------------------------------------------------------------------------")
print("                                     GRAPH 1                                                 ")
print("---------------------------------------------------------------------------------------------")

print("The eigenvalue and eigenvector of M of the first graph approximated with power method are:\n"+
"Eigenvalue = ", g1_lamda, "\n" +
"Eigenvector = ", g1_vector,"\n")
print("The sum of elements of the eigenvector is = ", sum(g1_vector), "\n")
print("RANK OF NODES: \n")

ranks_1 =len(g1_vector) - st.rankdata(g1_vector) + 1 # [1.0, 4.0, 2.0, 3.0]
ranks_1 = [int(i) for i in ranks_1] # [1, 4, 2, 3]

indexes = [i + 1 for i in range(len(ranks_1))] # creating the indexes for zipping

dictRanks_1 = dict(zip(ranks_1,indexes)) # putting ranks and nodes in a dictionary 
dictRanks_1 = dict(sorted(dictRanks_1.items())) # ranks on the left and nodes on the right 

for rank, node in dictRanks_1.items():
    print("Rank ", rank," ---> ", node)


print("---------------------------------------------------------------------------------------------")
print("                                     GRAPH 2                                                 ")
print("---------------------------------------------------------------------------------------------")

print("The eigenvalue and eigenvector of M of the second graph approximated with power method are:\n"+
"Eigenvalue = ", g2_lamda, "\n" +
"Eigenvector = ", g2_vector, "\n")
print("The sum of elements of the eigenvector is = ", sum(g2_vector))
print("RANK OF NODES: \n")

ranks_2 =len(g2_vector) - st.rankdata(g2_vector) + 1
ranks_2 = [int(i) for i in ranks_2] 

indexes = [i + 1 for i in range(len(ranks_2))] 

dictRanks_2 = dict(zip(ranks_2,indexes)) 
dictRanks_2 = dict(sorted(dictRanks_2.items()))  

for rank, node in dictRanks_2.items():
    print("Rank ", rank," ---> ", node)

print("---------------------------------------------------------------------------------------------")
print("                                     GRAPH 3                                                 ")
print("---------------------------------------------------------------------------------------------")

print("The eigenvalue and eigenvector of M of the third graph approximated with power method are:\n"+
"Eigenvalue = ", g3_lamda, "\n" +
"Eigenvector = ", g3_vector, "\n")
print("The sum of elements of the eigenvector is = ", sum(g3_vector))

print("RANK OF NODES: \n")

ranks_3 =len(g3_vector) - st.rankdata(g3_vector) + 1 
ranks_3 = [int(i) for i in ranks_3] 

indexes = [i + 1 for i in range(len(ranks_3))] 

dictRanks_3 = dict(zip(ranks_3,indexes)) 
dictRanks_3 = dict(sorted(dictRanks_3.items())) 

for rank, node in dictRanks_3.items():
    print("Rank ", rank," ---> ", node)

print("---------------------------------------------------------------------------------------------")
print("                                     THE END                                                 ")
print("---------------------------------------------------------------------------------------------\n")
