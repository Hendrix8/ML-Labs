# Christos Panourgias
# AM: 2405

from curses import raw
import numpy as np
from numpy import linalg as la 
import networkx as nx
from scipy import stats as st
from tool import Tool
import matplotlib.pyplot as plt

tool = Tool() # instance of the class Tool 

# opening the files 
g1File = open("graph0.txt", "r")
g2File = open("graph1.txt", "r")
g3File = open("graph2.txt", "r")

# turning the files to lists
g1 = tool.txtToGraph(g1File)
g2 = tool.txtToGraph(g2File)
g3 = tool.txtToGraph(g3File)

# turning the lists to graphs 
G1 = nx.DiGraph(g1)
G2 = nx.DiGraph(g2)
G3 = nx.DiGraph(g3)


# sorting the pagerank dictionaries in respect to their keys
r1 = dict(sorted(nx.pagerank(G1).items()))
r2 = dict(sorted(nx.pagerank(G2).items()))
r3 = dict(sorted(nx.pagerank(G3).items()))


# storing the "clean" ranks in dictionaries {1: 2, 2:3, 4:1 }... etc
pgRank_1 = tool.toNumberRanks(r1)
pgRank_2 = tool.toNumberRanks(r2)
pgRank_3 = tool.toNumberRanks(r3)



print("\n---------------------------------------------------------------------")
print("                            GRAPH 1                                  ")
print("---------------------------------------------------------------------")

for rank, node in pgRank_1.items():
    print("Rank", rank," ===> ", node)

nx.draw(G1, with_labels = True) 
plt.show()


print("---------------------------------------------------------------------")
print("                            GRAPH 2                                  ")
print("---------------------------------------------------------------------")


for rank, node in pgRank_2.items():
    print("Rank", rank," ===> ", node)

nx.draw(G2, with_labels = True) 
plt.show()


print("---------------------------------------------------------------------")
print("                            GRAPH 3                                  ")
print("---------------------------------------------------------------------")


for rank, node in pgRank_3.items():
    print("Rank", rank," ===> ", node)

nx.draw(G3, with_labels = True) 
plt.show()


print("---------------------------------------------------------------------")
print("                            THE END                                  ")
print("---------------------------------------------------------------------")




