#Christos Panourgias 2405

import pandas as pd
import numpy as np 
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# NAIVE BAYES
train = pd.read_table("spam_train.txt", header=None) # reading the file
test = pd.read_table("spam_test.txt", header=None) # reading the file

new_names = {0 : "Type", 1 : "Text"} # new names for the headers
train.rename(columns=new_names, inplace=True) # renaming the headers
test.rename(columns=new_names, inplace=True) # renaming the headers

vectorizer = CountVectorizer() # creating a CountVectorizer object
naive_Bayes = MultinomialNB(alpha=1.0) # creating our Naive Bayes Classifier , also setting alpha = 1.0 (for the model to do Laplace smoothing)

counts = vectorizer.fit_transform(train["Text"].values) # this splits each message into a list of words 
                                            #transforms them into numbers and counts occurences of words
targets = train["Type"].values # this contains all the classes of the messages

naive_Bayes.fit(counts, targets) # training our classifier

# ACCURACY FOR NAIVE BAYES
counts_test = vectorizer.transform(test["Text"].values) 

real_values = test["Type"].values # the real values of the test set 
pred_NB = naive_Bayes.predict(counts_test) # predictions of our trained classifier

accuracy_NB = sum([1 for i in range(len(real_values)) if real_values[i] == pred_NB[i]]) / len(real_values)

print("------------------------ NAIVE BAYES WITH LAPLACE SMOOTHING ------------------------")
print("The accuracy of the Naive Bayes model using Laplace smoothing is {0:.1f}% ".format(accuracy_NB * 100))
print("------------------------------------------------------------------------------------")
# SVM 

supVM = svm.SVC(kernel="rbf") # creating a support vector machine object using Gaussian Kernel

acc_gamma = {} # this dictionary will have gamma as keys and the respective accuracies as values

# gamma parameter can be seen as the inverse of the radius of influence, so low gamma means high influence and high gamma means low influence
for g in np.arange(0.01,1,0.05):
    
    supVM.gamma = g # changing the radius

    supVM.fit(counts, targets) # training the model with this radius 

    # finding the accuracy of the model
    pred_SVM = supVM.predict(counts_test) 
    accuracy_SVM = sum([1 for i in range(len(real_values)) if real_values[i] == pred_SVM[i]]) / len(real_values)

    acc_gamma[round(g,2)] = round(accuracy_SVM,2)

print("------------------------ SVM ACCURACY FOR DIFFERENT GAMMA ------------------------")
for i in range(len(acc_gamma)):
    print("Gamma = {} : Accuracy {}".format(list(acc_gamma.keys())[i], list(acc_gamma.values())[i]))
print("----------------------------------------------------------------------------------")

print("--------------------------------- BEST GAMMA -------------------------------------")
print("The accuracy of the SVM model using Gaussian kernel and the best gamma is 98" + "%" + " and \n the best gamma is 0.01")
print("----------------------------------------------------------------------------------")


