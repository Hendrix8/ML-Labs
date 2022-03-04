# Christos Panourgias
# AM: 2405

import numpy as np
from numpy import linalg as la 
import networkx as nx
import scipy as sp
from tool import Tool
import matplotlib.pyplot as plt

tool = Tool()

g1File = open("graph0.txt", "r")

g1 = tool.txtToGraph(g1File)
G1 = nx.DiGraph(g1)

rank_1 = nx.pagerank(G1)
print(rank_1[1])


'''nx.draw(G1, with_labels = True) 
plt.show()'''

'''print("---------------------------------------------------------------------")
print("                            GRAPH 1                                  ")
print("---------------------------------------------------------------------")'''


'''nx.draw(G1, with_labels=True)
plt.show()'''

'''print("---------------------------------------------------------------------")
print("                            GRAPH 2                                  ")
print("---------------------------------------------------------------------")

nx.draw(G2, with_labels=True)
plt.title("GRAPH 2")
plt.show()

print(nx.pagerank(G2,d)) # printing the ranks


print("---------------------------------------------------------------------")
print("                            GRAPH 3                                  ")
print("---------------------------------------------------------------------")

nx.draw(G3, with_labels=True)
plt.title("GRAPH 3")
plt.show()

print(nx.pagerank(G3,d)) # printing the ranks'''
