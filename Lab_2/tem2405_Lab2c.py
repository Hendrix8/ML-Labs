import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from mlTool import tool

# opening the files 
set1_train = open("Data/set1_train.txt","r")
set2_train = open("Data/set2_train.txt","r")
set1_test = open("Data/set1_test.txt","r")
set2_test = open("Data/set2_test.txt","r")

# creating an instance of the tool class 
tool = tool()

# extracting the normalized data 
x1, y1 = tool.LGextractData_txt(set1_train)
x2, y2 = tool.LGextractData_txt(set2_train)
x1_test, y1_test = tool.LGextractData_txt(set1_test)
x2_test, y2_test = tool.LGextractData_txt(set2_test)

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


# initializing theta to be 0 and decision point to be 0.5
theta = np.zeros(len(x1[0]))
decision_point = 0.5

# using newton to estimate theta
#theta, Jtheta = tool.newton(x1, y1, theta, J, Jp, Jpp)
#print(theta)
#print(Jtheta)

theta = np.array([-0.69624563, -1.11116892, -3.31398542]) # for this theta J'(theta) = [-0.00161076 -0.01000832 -0.00115679]
log_reg = LogisticRegression()

# removing the 1 from 0 position (sk learn does not need it)
sk_x1 = [i[1::] for i in x1]
log_reg.fit(sk_x1, y1)

sk_theta_1 = log_reg.coef_[0].tolist() # turning sk_theta to list so that I can add the constant term
sk_theta_1.insert(0,log_reg.intercept_[0]) # adding the constant term
sk_theta_1 = np.array(sk_theta_1) # making sk_theta a np.array again

sk_x2 = [i[1::] for i in x2]
log_reg.fit(sk_x2, y2)

sk_theta_2 = log_reg.coef_[0].tolist() # turning sk_theta to list so that I can add the constant term
sk_theta_2.insert(0,log_reg.intercept_[0]) # adding the constant term
sk_theta_2 = np.array(sk_theta_2) # making sk_theta a np.array again

print("------------------------------------------------------------------------------------------------------")
print("                            PROBABILITY PREDICTIONS FOR TEST SET 1                                    ")
print("------------------------------------------------------------------------------------------------------")

for i in range(len(x1_test)):
    print("Probability for (x1, x2) = (", x1_test[i][1],",",x1_test[i][2], ") is : ",\
        round(h(x1_test[i], theta), 4))

y1_pred = [0 if i < decision_point else 1 for i in [h(x1_test[i], theta) for i in range(len(x1_test))]]

print("------------------------------------------------------------------------------------------------------")
print("                                 CONFUSION MATRIX FOR Y1_PRED                                      ")
print("------------------------------------------------------------------------------------------------------")




#     -----------------
#    |   TP   |   FP   |    This is the confusion matrix which gives evaluation on our model, where
#     -----------------          TP = True-Positive, FP = False-Positive
#    |   FN   |   TN   |         FN = False-Negative, TN = True-Negative
#     -----------------           




CM1 = confusion_matrix(y1_test, y1_pred) 

print("                                   ------------------------")
print("                                   | TP = ", CM1[0][0], "| FP = ", CM1[0][1], "|")
print("                                   ------------------------")
print("                                   | FN = ", CM1[1][0], "| TN = ", CM1[1][1], "|")
print("                                   ------------------------")


print("------------------------------------------------------------------------------------------------------")
print("                                           THE END                                                    ")
print("------------------------------------------------------------------------------------------------------")

print("------------------------------------------------------------------------------------------------------")
print("                         PROBABILITY SK-LEARN PREDICTIONS FOR TEST SET 1                              ")
print("------------------------------------------------------------------------------------------------------")

for i in range(len(x1_test)):
    print("Probability for (x1, x2) = (", x1_test[i][1],",",x1_test[i][2], ") is : ",\
        round(h(x1_test[i], sk_theta_1), 4))

sk_x1_test = [i[1::] for i in x1_test] # removing 1 from position 0 
y1_sk_pred = log_reg.predict(sk_x1_test)

print("------------------------------------------------------------------------------------------------------")
print("                                 CONFUSION MATRIX FOR Y1_SK_PRED                                      ")
print("------------------------------------------------------------------------------------------------------")

CM_sk1 = confusion_matrix(y1_test, y1_sk_pred)


print("                                   ------------------------")
print("                                   | TP = ", CM_sk1[0][0], "| FP = ", CM_sk1[0][1], "|")
print("                                   ------------------------")
print("                                   | FN = ", CM_sk1[1][0], "| TN = ", CM_sk1[1][1], "|")
print("                                   ------------------------")



print("------------------------------------------------------------------------------------------------------")
print("                                           THE END                                                    ")
print("------------------------------------------------------------------------------------------------------")


print("------------------------------------------------------------------------------------------------------")
print("                            PROBABILITY PREDICTIONS FOR TEST SET 2                                    ")
print("------------------------------------------------------------------------------------------------------")

for i in range(len(x2_test)):
    print("Probability for (x1, x2) = (", x2_test[i][1],",",x2_test[i][2], ") is : ",\
        round(h(x2_test[i], theta), 4))


y2_pred = [0 if i < decision_point else 1 for i in [h(x2_test[i], theta) for i in range(len(x2_test))]]

print("------------------------------------------------------------------------------------------------------")
print("                                 CONFUSION MATRIX FOR Y2_PRED                                      ")
print("------------------------------------------------------------------------------------------------------")

CM2 = confusion_matrix(y2_test, y2_pred)

print("                                   ------------------------")
print("                                   | TP = ", CM2[0][0], "| FP = ", CM2[0][1], "|")
print("                                   ------------------------")
print("                                   | FN = ", CM2[1][0], "| TN = ", CM2[1][1], "|")
print("                                   ------------------------")



print("------------------------------------------------------------------------------------------------------")
print("                                           THE END                                                    ")
print("------------------------------------------------------------------------------------------------------")

print("------------------------------------------------------------------------------------------------------")
print("                       PROBABILITY SK-LEARN PREDICTIONS FOR TEST SET 2                                ")
print("------------------------------------------------------------------------------------------------------")

for i in range(len(x2_test)):
    print("Probability for (x1, x2) = (", x2_test[i][1],",",x2_test[i][2], ") is : ",\
        round(h(x2_test[i], sk_theta_2), 4))

sk_x2_test = [i[1::] for i in x2_test] # removing 1 from position 0 
y2_sk_pred = log_reg.predict(sk_x2_test)

print("------------------------------------------------------------------------------------------------------")
print("                                 CONFUSION MATRIX FOR Y2_SK_PRED                                      ")
print("------------------------------------------------------------------------------------------------------")

CM_sk2 = confusion_matrix(y2_test, y2_sk_pred)
print("                                   ------------------------")
print("                                   | TP = ", CM_sk2[0][0], "| FP = ", CM_sk2[0][1], "|")
print("                                   ------------------------")
print("                                   | FN = ", CM_sk2[1][0], "| TN = ", CM_sk2[1][1], "|")
print("                                   ------------------------")

print("------------------------------------------------------------------------------------------------------")
print("                                           THE END                                                    ")
print("------------------------------------------------------------------------------------------------------")

tool.scatterClasses(x1, y1, yllim=-2.5, yhlim=17, xllim=-4, xhlim=3.5,title="x = (x1,x2) for the train_set1", \
                    xlabel="x1", ylabel="x2")
tool.scatterClasses(x1_test, y1_test, yllim=-2.5, yhlim=17, xllim=-4, xhlim=3.5, title="x = (x1,x2) for the test_set1", \
                    xlabel="x1", ylabel="x2")
tool.scatterClasses(x2, y2, yllim=-3.5, yhlim=2.8, xllim=-4, xhlim=3.5, title="x = (x1,x2) for the train_set2", \
                    xlabel="x1", ylabel="x2")
tool.scatterClasses(x2_test, y2_test, yllim=-3.5, yhlim=2.8, xllim=-4, xhlim=3.5, title="x = (x1,x2) for the test_set2", \
                    xlabel="x1", ylabel="x2")
