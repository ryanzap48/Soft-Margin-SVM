import numpy as np
from soft_svm import *
from sklearn.datasets import make_classification

n_samples = 200
X, y = make_classification(n_samples=n_samples, n_features=4, n_informative=4, n_redundant=0, n_clusters_per_class=1, random_state=1)
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = 2*y - 1
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

C = 1000
alpha = 0.005
n_epoch = 1000

w = train(Xtrain, Ytrain, C, alpha, n_epoch)

yhat_train = predict(Xtrain, w)
yhat_test = predict(Xtest, w)
test_loss = compute_J(Xtest, Ytest, w, C)

accuracy_train = np.mean(yhat_train == Ytrain)
accuracy_test = np.mean(yhat_test == Ytest)

with open("report.txt", "w") as f:
    f.write(f"C: {C}\n")
    f.write(f"alpha: {alpha}\n")
    f.write(f"n_epoch: {n_epoch}\n")
    f.write(f"Training Accuracy: {accuracy_train}\n")
    f.write(f"Test Accuracy: {accuracy_test}\n")
    f.write(f"Test Objective Value: {test_loss}\n")

"""
C: 1000
alpha: 0.005
n_epoch: 1000
Training Accuracy: 0.93
Test Accuracy: 0.92
Test Objective Value: 600106.2863511132
"""
