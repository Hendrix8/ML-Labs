import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn import linear_model
from mlTool import tool

# reading files 
train = open("Data/f_train.txt", "r")
test = open("Data/f_test.txt", "r")

# creating an object of my class
tool = tool()

# extracting the data 
x, y = tool.extractData_txt(train)
x_test, y_test = tool.extractData_txt(test)


def phi(x):
    return np.array([1, x, x**2, x**3])

# this is our x data transformed by phi (the phi(x) returns an array with 1 on the first column that is why we dont take it here)
x_phi = [phi_x for phi_x in [phi(i[1]) for i in x] ] # a list that contains all data transformations in np.arrays

# h(x;theta ) is the same as in ex1
def h(x, theta):
    return np.dot(theta.T, x)

# Cost function J(theta) (the same as in ex1)
def J(x, y, h, theta):
    n = len(x)
    return  1.0/(2*n) * sum( [ (h(x[i], theta) - y[i])**2 for i in range(len(x))])

def Jprime(x, y, h,  theta):
    n = len(x)
    return 1.0/n * sum( [ (h(x[i], theta) - y[i]) * x[i] for i in range(len(x))])

# we have a mulitvariate linear regression just as we had in ex1
lrate = 0.4
theta = np.ones(4)
theta, iter, Jtheta = tool.GD(x_phi, y, J, Jprime, h, theta, lrate)

# computing Etheta 
x_phi_test = [phi_x for phi_x in [phi(i[1]) for i in x_test] ] # first computing the tranformation of x_test through phi
Etheta = la.norm([h(x_phi_test[i], theta) - y_test[i] for i in range(len(x_phi_test))], 2)


# finding the answer with sk-learn
reg = linear_model.LinearRegression() # Creating linear regression class object
sk_x = [i[1::] for i in x_phi] # deleting the first feature which is 1 (sk_learn does not need it)
reg.fit(sk_x, y) # fitting the line

sk_theta = reg.coef_.tolist() # turning sk_theta to list so that I can add the constant term
sk_theta.insert(0,reg.intercept_) # adding the constant term
sk_theta = np.array(sk_theta) # making sk_theta a np.array again

# calculating Etheta that is produced by the sk_learn linear regression algorithm
sk_Etheta = la.norm([h(x_phi_test[i], sk_theta) - y_test[i] for i in range(len(x_phi_test))], 2)


print("-------------------------------------------------------------------------------")
print("                                RESULTS                                        ")
print("-------------------------------------------------------------------------------")
print("Theta = " + np.array2string(theta) + "\nSk-Learn Theta = " +str(sk_theta) + \
      "\nLearning rate = 0.4 \nIterations = " + str(iter) + "\nEpsilon = " + str(10**(-3)) + \
      "\nDelta = " + str(10**(-3)) + "\nEpsilon Theta Error = " + str(Etheta) + \
      "\nSk-Learn Epsilon Theta Error = " + str(sk_Etheta))

print("-------------------------------------------------------------------------------")
print("                                THE END                                        ")
print("-------------------------------------------------------------------------------")

xaxis = np.linspace(0,len(x), len(x)) # creating an x axis for the graph
#xaxis_h = np.linspace(0,len(x_phi_test), len(x_phi_test))
plt.scatter(xaxis, y_test)
#plt.plot(xaxis_h, [h(x_phi_test[i]) for i in range(len(x_phi_test))])
plt.show()