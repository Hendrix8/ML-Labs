import numpy as np 
import numpy.linalg as la

# this class contains the tools that I will use to solve the exercises
class tool():

    def __init__(self):
        pass
    
    def extractData_txt(self, file):
        x = [] 
        y = []
        for line in file.readlines()[1::] : # ignoring the first line with the names of the features

            # creating an with all the floats of each feature and adding them to x
            x.append([float(i) for i in line.split()][0:-1])
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

    # implementing the gradient decent algorithm 
    # J is the cost function and Jprime is the derivative of the cost function
    # h is the hypothesis function and theta is the parameters
    # x and y are the training data
    # lrate is the learning rate 
    # delta is the threshold for J and epsilon is the threshold for the 1-norm of theta(n+1)-theta(n)
    # returns theta, iterations and a list of J(theta) values throughout the GD algorithm
    def GD(self,x, y, J, Jprime, h, theta, lrate=0.01, max_iter = 10000, epsilon=10**(-3), delta=10**(-3)):

        Jtheta = [] # list that contains all outputs of J function for all inputs of theta
        iter = 0
        theta_old = np.random.randint(1,10,len(theta))
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