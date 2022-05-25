#Christos Panourgias 2405

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

train_img = pd.read_csv("images_train.csv",header=None)
test_img = pd.read_csv("images_test.csv", header=None)
train_labels = pd.read_csv("labels_train.csv", header=None)
test_labels = pd.read_csv("labels_test.csv", header=None)

# Normalizing the data 
train_img = np.array(train_img)
train_img = train_img / train_img.max()

test_img = np.array(test_img)
test_img = test_img / test_img.max()


# Currently, MLPClassifier supports only the Cross-Entropy loss function,
# which allows probability estimates by running the predict_proba method.
# MLPClassifier supports multi-class 
# classification by applying Softmax as the output function.

print("---------------------- NEURAL NETWORK WITHOUT SMOOTHING FACTOR ----------------------")
print("  --------------------------- LOSS AND ITERATIONS --------------------------------")
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic",\
solver="sgd", learning_rate_init=0.6, verbose=True)

mlp.fit(train_img, train_labels)

real_values = test_labels.values
pred_values = mlp.predict(test_img)

print("  --------------------------------------------------------------------------------  ")
print("  ---------------------------------- ACCURACY ------------------------------------  ")
print("The accuracy of the neural network without a smoothing factor is : {} ".format(mlp.score(test_img, test_labels)))
print("-------------------------------------------------------------------------------------")

# plotting the loss 
plt.plot(mlp.loss_curve_)
plt.title("Loss Function without Smoothing Factor")
plt.xlabel("Iterations")
plt.ylabel("Loss Function")
plt.show()

print("------------------------ NEURAL NETWORK WITH SMOOTHING FACTOR ------------------------")
print("  ----------------------------- LOSS AND ITERATIONS ----------------------------------")

mlp_SF = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic",\
solver="sgd", learning_rate_init=0.6, verbose=True, alpha = 1e-10)

mlp_SF.fit(train_img, train_labels)

real_values = test_labels.values
pred_values = mlp.predict(test_img)

print("  ----------------------------------------------------------------------------------  ")
print("  ------------------------------------ ACCURACY -------------------------------------  ")
print("The accuracy of the neural network without a smoothing factor is : {} ".format(mlp_SF.score(test_img, test_labels)))
print("-------------------------------------------------------------------------------------")

# plotting the loss 
plt.plot(mlp_SF.loss_curve_)
plt.title("Loss Function with Smoothing Factor")
plt.xlabel("Iterations")
plt.ylabel("Loss Function")
plt.show()


