import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn import linear_model
from mlTool import tool

# reading files 
train = open("Data/f_train.txt", "r")
test = open("Data/f_test.txt", "r")
small_train = open("Data/f_small.txt","r")

# creating an object of my class
tool = tool()

# extracting the data 
x, y = tool.extractData_txt(train)
x_test, y_test = tool.extractData_txt(test)
x_small, y_small = tool.extractData_txt(small_train)

# x is the feature list and k is the 1 + x + x^2 +...+ x^k array that it returns
def phi(x, k):
    phi_x = [x ** i for i in range(0, k)]
    return np.array(phi_x)

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
# this is our x data transformed by phi (the phi(x) returns an array with 1 on the first column that is why we dont take it here)
x_phi = [phi_x for phi_x in [phi(i[1], 4) for i in x] ] # a list that contains all data transformations in np.arrays

lrate = 0.4 # best learning rate after expirementing
theta = np.ones(4) # starting theta 
theta, iter, Jtheta = tool.GD(x_phi, y, J, Jprime, h, theta, lrate) # results using my GD implementation

# computing Etheta 
x_phi_test = [phi_x for phi_x in [phi(i[1], 4) for i in x_test] ] # first computing the tranformation of x_test through phi
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

xaxis_test = np.linspace(0,len(y_test), len(y_test)) # creating a x axis for the graph
plt.title("h(theta) and h(sk_theta) for test set.")
plt.scatter(xaxis_test, y_test)
plt.plot(xaxis_test, [h(x_phi_test[i], theta) for i in range(len(x_phi_test))], label="h(theta)", color="b")
plt.plot(xaxis_test, [h(x_phi_test[i], sk_theta) for i in range(len(x_phi_test))], label="h(sk_theta)", color="r")
plt.show()

xaxis_train = np.linspace(0,len(y), len(y)) # creating a x axis for the graph
plt.title("h(theta) and h(sk_theta) for train set.")
plt.scatter(xaxis_train, y)
plt.plot(xaxis_train, [h(x_phi[i], theta) for i in range(len(x_phi))], label="h(theta)", color="b")
plt.plot(xaxis_train, [h(x_phi[i], sk_theta) for i in range(len(x_phi))], label="h(sk_theta)", color="r")
plt.show()


