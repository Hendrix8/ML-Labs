import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn import linear_model
from mlTool import tool

# opening the files 
train = open("Data/car_train.txt", "r")
test = open("Data/car_test.txt", "r")


# initializing variables
tool = tool()
theta = np.ones(5)

# extracting the data using the function created in the mlTool.tool class
x, y = tool.LGextractData_txt(train) 
x_test, y_test = tool.LGextractData_txt(test)

# defining the hypothesis function for the linear model 
def h(x, theta):
    return np.dot(theta.T, x)

# defining the cost function
def J(x, y, h, theta):
    n = len(x)
    return  1.0/(2*n) * sum( [ (h(x[i], theta) - y[i])**2 for i in range(n)])

# defining the derivative of the cost function 
def Jprime(x, y, h,  theta):
    n = len(x)
    return 1.0/n * sum( [ (h(x[i], theta) - y[i]) * x[i] for i in range(n)])


# Linear Regression with sk_learn
reg = linear_model.LinearRegression() # Creating linear regression class object
sk_x = [i[1::] for i in x] # deleting the first feature which is 1 (sk_learn does not need it)
reg.fit(sk_x, y) # fitting the line

sk_theta = reg.coef_.tolist() # turning sk_theta to list so that I can add the constant term
sk_theta.insert(0,reg.intercept_) # adding the constant term
sk_theta = np.array(sk_theta) # making sk_theta a np.array again

# calculating Etheta that is produced by the sk_learn linear regression algorithm
sk_Etheta = la.norm([h(x_test[i], sk_theta) - y_test[i] for i in range(len(x_test))], 2)

# calculating theta using my model of Gradient Decent
lrate = 0.53 # choosing lrate = 0.53 after experimenting with the convergence of GD
theta, iter, Jtheta = tool.GD(x, y, J, Jprime, h, theta, lrate)

# Calculating the Epsilon theta error of my model
Etheta = la.norm([h(x_test[i], theta) - y_test[i] for i in range(len(x_test))], 2)

# PROBLEM (a) : Printing the important values 
print("-------------------------------------------------------------------------------")
print("                                RESULTS                                        ")
print("-------------------------------------------------------------------------------")
print("Theta = " + np.array2string(theta) + "\nSk-Learn Theta = " +str(sk_theta) + \
      "\nLearning rate = " + str(lrate) + "\nIterations = " + str(iter) + "\nEpsilon = " + str(10**(-3)) + \
      "\nDelta = " + str(10**(-3)) + "\nEpsilon Theta Error = " + str(Etheta) + \
      "\nSk-Learn Epsilon Theta Error = " + str(sk_Etheta))

print("-------------------------------------------------------------------------------")
print("                                THE END                                        ")
print("-------------------------------------------------------------------------------")

# PROBLEM (b) showing the progress of J(theta) thoughout the Gradient Decent process
xaxis = np.linspace(0,len(Jtheta),len(Jtheta))
plt.title("J(theta) progress throughout Gradient Decent Algorithm")
plt.xlabel("Iterations")
plt.ylabel("J(theta)")
plt.plot(xaxis, Jtheta)
#plt.show()