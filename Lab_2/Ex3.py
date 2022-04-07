import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn import linear_model
from mlTool import tool

# opening the files 
set1_train = open("Data/set1_train.txt","r")
set2_train = open("Data/set2_train.txt","r")
set1_test = open("Data/set1_test.txt","r")
set2_test = open("Data/set2_test.txt","r")

# creating an instance of the tool class 
tool = tool()

# extracting the normalized data 
x1, y1 = tool.extractData_txt(set1_train)
x2, y2 = tool.extractData_txt(set2_train)
x1_test, y1_test = tool.extractData_txt(set1_test)
x2_test, y2_test = tool.extractData_txt(set2_test)

# creating the hypothesis function for logistic regression 
def h(x, theta):
    thetaT_x = np.dot(theta.T, x)
    return 1.0/(1.0+np.exp(-thetaT_x))

# function of cost 
def J(x, y, theta):
    n = len(x)
    return 1.0/n * sum([ y[i] * np.log(h(x[i], theta)) + (1 - y[i]) * np.log(1 - h(x[i], theta))\
                        for i in range(n)])

# first derivative of J(theta)
def Jp(x, y, theta):
    n = len(x)
    thetaT_x = [np.dot(theta.T, x[i]) for i in range(n)]
    return 1.0/n * sum([ (-y[i]) * (x[i] * np.exp(thetaT_x[i])) / (1 + np.exp(thetaT_x[i])) + \
                         (1 - y[i]) * (x[i] - (x[i] * np.exp(thetaT_x[i])) / (1 + np.exp(thetaT_x[i]))) \
                            for i in range(n)])

# second derivative o J(theta)
def Jpp(x, y, theta):
    n = len(x)
    thetaT_x = [np.dot(theta.T, x[i]) for i in range(n)]
    
    return 1.0/n * sum( [ ( (-y[i] * x[i]**2) * np.exp(thetaT_x[i]) * (1 + np.exp(thetaT_x[i])) - x[i] * np.exp(2 * thetaT_x[i]) ) / (1 + np.exp(thetaT_x[i])) ** 2 + \
            (1 - y[i]) * (x[i] - (x[i]**2 * np.exp(thetaT_x[i]) * (1 + np.exp(thetaT_x[i])) - x[i] * np.exp(2 * thetaT_x[i]) / (1 + np.exp(thetaT_x[i])) ** 2)) \
                for i in range(n) ])


# initializing theta to be 0 
theta = np.zeros(len(x1[0]))

theta = tool.newton(x1, y1, theta, Jp, Jpp)
print(theta)