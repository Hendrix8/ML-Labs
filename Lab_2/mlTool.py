from cProfile import label
from turtle import color
import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt

# this class contains the tools that I will use to solve the exercises
class tool():

    def __init__(self):
        pass
    
    # extracts data with 1 at 0 position ready for linear regression
    def LGextractData_txt(self, file):
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
        
        maxy = max([abs(i) for i in y])
        y = [i/maxy for i in y]

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
    def GD(self,x, y, J, Jprime, h, theta,\
        lrate=0.01, max_iter = 10000, epsilon=10**(-3), delta=10**(-3)):

        Jtheta = [] # list that contains all outputs of J function for all inputs of theta
        iter_ = 0
        theta_old = np.random.randint(1,10,len(theta))
        while J(x, y, h, theta) > delta and la.norm(theta - theta_old, 1) > epsilon and iter_ <= max_iter:

            # Updating theta and keeping a copy of the old theta to compute the norm in the condition of the while loop
            theta_old = theta
            theta = theta - lrate * Jprime(x, y, h, theta)
            Jtheta.append(J(x, y, h, theta))
            iter_ += 1

            #print(J(x, y, h, theta), la.norm(theta-theta_old), iter_) # this is for watching how J(theta) is being minimized and therefore determine the learning rate
            
            if iter_ == max_iter:
                print("-----------------------------------")
                print("Gradient Decent does not converge.")
                print("-----------------------------------")
                break

        return theta, iter_, Jtheta
    
    # we want to find where the Jp function is 0 using the newton method
    def newton(self, x, y, theta,J, Jp, Jpp, epsilon=10**(-4), max_iter=10000):
        
        iter_ = 0
        Jtheta = []
        theta_old = np.random.randint(1,10,len(theta))
        while la.norm(theta_old - theta, 1) > epsilon and iter_ <= max_iter:

            theta_old = theta
            theta = theta + Jp(x, y, theta) / Jpp(x, y, theta) # it is plus because you want to maximize
            iter_ += 1

            print(Jp(x, y, theta), theta, J(x, y, theta), la.norm(theta_old - theta,1), iter_ ) # this helps on the implementation (to see how the results go throughout the Newton method)
            Jtheta.append(J(x, y, theta))
            if iter_ == max_iter:
                print("Newtons method Does not converge.")
        
        return theta, Jtheta


    # creates a scatter plot for x1,x2 where x = [ (1,x1,x2)_1,....,(1,x1,x2)_n ] and y = [0, 1, 1, 0 ,1 ... , 1, 1]
    def scatterClasses(self, x, y, title=None, xlabel=None, ylabel=None, boundary_line=False,yllim=0, yhlim=None, xllim=0, xhlim=None, xline=None, yline=None):

        x_0 = [x[i][1::] for i in range(len(x)) if y[i] == 0 ] # taking only the samples from class 0 from the set_train extracting the 1 at position 0 
        #y_0 = [i for i in y1 if i == 0.0] # taking the y = 0 from the set_train

        x_0_X = [i[0] for i in x_0] # taking the x1 from the class 0 of the set_train set
        x_0_Y = [i[1] for i in x_0] # taking the x2 from the class 0 of the set_train set 

        x_1 = [x[i][1::] for i in range(len(x)) if y[i] == 1 ] # taking only the samples from class 1 from the set_train extracting the 1 at position 0 
        #y1_1 = [i for i in y1 if i == 1.0] # taking the y = 0 from the set_train

        x_1_X = [i[0] for i in x_1] # taking the x1 from the class 1 of the set_train set
        x_1_Y = [i[1] for i in x_1] # taking the x2 from the class 1 of the set_train set 


        plt.scatter(x_0_X, x_0_Y, label="0 class", marker="+")
        plt.scatter(x_1_X, x_1_Y, label="1 class",marker="*")
        if boundary_line == True:
            plt.plot(xline, yline, label="Decision Line", color="r", linewidth=0.7)

        if title != None or xlabel != None or ylabel != None:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        if yhlim != None:
            plt.ylim(yllim, yhlim)
        if xhlim != None:
            plt.xlim(xllim, xhlim)

        plt.legend()
        plt.show()