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
    return  1.0/(2*n) * sum( [ (h(x[i], theta) - y[i])**2 for i in range(n)])

def Jprime(x, y, h,  theta):
    n = len(x)
    return 1.0/n * sum( [ (h(x[i], theta) - y[i]) * x[i] for i in range(n)])

# we have a mulitvariate linear regression just as we had in ex1
# this is our x data transformed by phi (the phi(x) returns an array with 1 on the first column that is why we dont take it here)
x_phi = [phi_x for phi_x in [phi(i[1], 4) for i in x] ] # a list that contains all data transformations in np.arrays

lrate = 0.43 # best learning rate after expirementing
theta = np.ones(4) # starting theta 
theta, iter_, Jtheta = tool.GD(x_phi, y, J, Jprime, h, theta, lrate) # results using my GD implementation

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
print("                         RESULTS FOR 1st PROBLEM                               ")
print("-------------------------------------------------------------------------------")
print("Theta = " + np.array2string(theta) + "\nSk-Learn Theta = " +str(sk_theta) + \
      "\nLearning rate = 0.43 \nIterations = " + str(iter_) + "\nEpsilon = " + str(10**(-3)) + \
      "\nDelta = " + str(10**(-3)) + "\nEpsilon Theta Error = " + str(Etheta) + \
      "\nSk-Learn Epsilon Theta Error = " + str(sk_Etheta))

print("-------------------------------------------------------------------------------")
print("                                THE END                                        ")
print("-------------------------------------------------------------------------------")

xaxis_test = np.linspace(0,len(y_test), len(y_test)) # creating a x axis for the graph
plt.title("h(theta) and h(sk_theta) for test set.")
plt.scatter(xaxis_test, y_test)
plt.xlabel("x")
plt.ylabel("h(x;theta) and h(x;sk_theta)")
plt.plot(xaxis_test, [h(x_phi_test[i], theta) for i in range(len(x_phi_test))], label="h(x;theta)", color="b")
plt.plot(xaxis_test, [h(x_phi_test[i], sk_theta) for i in range(len(x_phi_test))], label="h(x;sk_theta)", color="r")
plt.legend()
plt.show()

xaxis_train = np.linspace(0,len(y), len(y)) # creating a x axis for the graph
plt.title("h(theta) and h(sk_theta) for train set.")
plt.scatter(xaxis_train, y)
plt.xlabel("x")
plt.ylabel("h(x;theta) and h(x;sk_theta)")
plt.plot(xaxis_train, [h(x_phi[i], theta) for i in range(len(x_phi))], label="h(x;theta)", color="b")
plt.plot(xaxis_train, [h(x_phi[i], sk_theta) for i in range(len(x_phi))], label="h(x;sk_theta)", color="r")
plt.legend()
plt.show()

def LGsmall(x_small, y_small, k, lrate, max_iter=10000):
    x_small_phi = [phi_x_small for phi_x_small in [phi(x[1], k) for x in x_small]]
    theta_sm = np.ones(k) # starting theta 
    theta_sm, iter_sm, Jtheta_sm = tool.GD(x_small_phi, y_small, J, Jprime, h, theta_sm, lrate, max_iter=max_iter) # results using my GD implementation
    Etheta_sm = la.norm([h(x_small_phi[i], theta_sm) - y_small[i] for i in range(len(x_small_phi))], 2)

    print("-------------------------------------------------------------------------------")
    print("                                RESULTS FOR k = " +str(k)+ "                   ")
    print("-------------------------------------------------------------------------------")
    print("Theta = " + np.array2string(theta_sm) +\
        "\nLearning rate = "+ str(lrate) + "\nIterations = " + str(iter_sm) + "\nEpsilon = " + str(10**(-3)) + \
        "\nDelta = " + str(10**(-3)) + "\nEpsilon Theta Error = " + str(Etheta_sm))

    print("-------------------------------------------------------------------------------")
    print("                                THE END                                        ")
    print("-------------------------------------------------------------------------------")

    xaxis_small = np.linspace(0,len(y_small), len(y_small)) # creating a x axis for the graph
    plt.title("h(x;theta) for different k.")
    plt.xlabel("x")
    plt.ylabel("h(x;theta)")
    plt.scatter(xaxis_small, y_small)
    plt.plot(xaxis_small, [h(x_small_phi[i], theta_sm) for i in range(len(x_small_phi))], label="h(x;theta) for k = " + str(k))


def LGsmall_sk(x_small, y_small, k, max_iter=10000):

    x_small_phi = [phi_x_small for phi_x_small in [phi(x[1], k) for x in x_small]]

    # sk-learn for small set
    reg = linear_model.LinearRegression() # Creating linear regression class object
    sk_x_sm = [i[1::] for i in x_small_phi] # deleting the first feature which is 1 (sk_learn does not need it)
    reg.fit(sk_x_sm, y_small) # fitting the line

    sk_theta_sm = reg.coef_.tolist() # turning sk_theta to list so that I can add the constant term
    sk_theta_sm.insert(0,reg.intercept_) # adding the constant term
    sk_theta_sm = np.array(sk_theta_sm) # making sk_theta a np.array again

    sk_Etheta_sm = la.norm([h(x_small_phi[i], sk_theta_sm) - y_small[i] for i in range(len(x_small_phi))], 2)
    
    print("-------------------------------------------------------------------------------")
    print("                       SK-LEARN RESULTS FOR k = " +str(k)+ "                   ")
    print("-------------------------------------------------------------------------------")
    print("\nSk-Learn Theta = " +str(sk_theta_sm) + \
        "\nSk-Learn Epsilon Theta Error = " + str(sk_Etheta_sm))

    print("-------------------------------------------------------------------------------")
    print("                                THE END                                        ")
    print("-------------------------------------------------------------------------------")

    xaxis_small = np.linspace(0,len(y_small), len(y_small)) # creating a x axis for the graph
    plt.title("h(x;sk_theta) for different k.")
    plt.xlabel("x")
    plt.ylabel("h(x;sk_theta)")
    plt.scatter(xaxis_small, y_small)
    plt.plot(xaxis_small, [h(x_small_phi[i], sk_theta_sm) for i in range(len(x_small_phi))], label="h(x;sk_theta) k = "+ str(k))


# for k = 3 the best learning rate = 0.63 after expirementing
LGsmall(x_small, y_small, 3, lrate=0.63)

# for k = 9 the best learning rate = 0.09 ( according to the smallest J(theta))
LGsmall(x_small, y_small, 5, lrate=0.09)

# for k = 10 best l-rate = 0.00042057 (smallest J(theta))
LGsmall(x_small, y_small, 10, lrate=0.00042057, max_iter=110000)

# for k = 20 best l-rate = 0.000000001842 (smallest J(theta))
LGsmall(x_small, y_small, 20, lrate=0.000000001842) #TODO: find out how to make it converge

plt.ylim(-2,5) # setting the limits so that the results can be shown better
plt.legend(loc=1)
plt.show()

# SK - LEARN same process
LGsmall_sk(x_small, y_small, 3)

# for k = 9 the best learning rate = 0.09 ( according to the smallest J(theta))
LGsmall_sk(x_small, y_small, 5)

# for k = 10 best l-rate = 0.00042057 (smallest J(theta))
LGsmall_sk(x_small, y_small, 10)

# for k = 20 best l-rate = (smallest J(theta))
LGsmall_sk(x_small, y_small, 20)

plt.legend(loc="upper left")
plt.show()
