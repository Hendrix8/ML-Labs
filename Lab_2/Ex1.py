import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn import linear_model

# opening the files 
train = open("Data/car_train.txt", "r")
test = open("Data/car_test.txt", "r")

# this function is for extracting data from txt files
def extractData(file):
    x = [] 
    y = []
    for line in file.readlines()[1::] : # ignoring the first line with the names of the features

        # creating an with all the floats of each feature and adding them to x
        x.append([float(i) for i in line.split()][0:4])
        y.append([float(i) for i in line.split()][-1]) # this is a list of numbers

    # inserting 1 in the first element of the features
    for l in x:
        l.insert(0, 1.0)

    # normalizing the data so that gradient decent can converge easily

    xlen = len(x[0]) # getting the length of a feature vector (5)
    counter = 0
    # creating a list that contains the lists of columns of the dataset
    colsOfx = []
    while counter < xlen:
        cols = []
        for i in x:
            cols.append(i[counter])
        colsOfx.append(cols)

        counter += 1

    meansOfx = [np.mean(i) for i in colsOfx]
    stdOfx = [np.std(i) for i in colsOfx]

    # normalizing x
    counter = 1 # not including the ones in the first column
    while counter < xlen:
        for i in range(len(x)):
            x[i][counter] = (x[i][counter] - meansOfx[counter]) / stdOfx[counter]
        counter += 1
    

    y = [i/max(y) for i in y]

    # transforming x,y into list of np.arrays
    x = [np.array(i) for i in x]
    y = [np.array(i) for i in y]

    # closing the file
    file.close()

    # returning the data
    return x, y

# initializing variables
theta = np.ones(5)
x, y = extractData(train) 
x_test, y_test = extractData(test)

# defining the hypothesis function for the linear model 
def h(x, theta):
    return np.dot(theta.T, x)

# defining the cost function
def J(x, y, h, theta):
    n = len(x)
    return  1.0/(2*n) * sum( [ (h(x[i], theta) - y[i])**2 for i in range(len(x))])

# defining the derivative of the cost function 
def Jprime(x, y, h,  theta):
    n = len(x)
    return 1.0/n * sum( [ (h(x[i], theta) - y[i]) * x[i] for i in range(len(x))])


# implementing the gradient decent algorithm 
def GD(x, y, theta, max_iter = 10000):

    Jtheta = [] # list that contains all outputs of J function for all inputs of theta
    delta =  10**(-3)
    epsilon = 10**(-3)
    lrate = 0.09
    iter = 0
    theta_old = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    while J(x, y, h, theta) > delta and la.norm(theta - theta_old, 1) > epsilon and iter <= max_iter:

        # Updating theta and keeping a copy of the old theta to compute the norm in the condition of the while loop
        theta_old = theta
        theta = theta - lrate * Jprime(x, y, h, theta)
        Jtheta.append(J(x, y, h, theta))
        iter += 1

        #print(J(x, y, h, theta), la.norm(theta-theta_old), iter) # this is for watching how J(theta) is being minimized and therefore determine the learning rate
        
        if iter == max_iter:
            print("-----------------------------------")
            print("Gradient Decent does not converge.")
            print("-----------------------------------")
            break

    return theta, iter, Jtheta

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
theta, iter, Jtheta = GD(x, y, theta)
#theta = np.array([  0.98151807, -0.21509956, -0.55496561, -0.03537969, -0.06947682])

# Calculating the Epsilon theta error of my model
Etheta = la.norm([h(x_test[i], theta) - y_test[i] for i in range(len(x_test))], 2)

# PROBLEM (a) : Printing the important values 
print("-------------------------------------------------------------------------------")
print("                                RESULTS                                        ")
print("-------------------------------------------------------------------------------")
print("Theta = " + np.array2string(theta) + "\nSk-Learn Theta = " +str(sk_theta) + \
      "\nLearning rate = 0.09 \nIterations = " + str(iter) + "\nEpsilon = " + str(10**(-3)) + \
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
plt.show()