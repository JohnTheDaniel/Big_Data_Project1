# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:54:03 2019

@author: yang_
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

X_train = np.loadtxt("Dataset/train/X_train.txt")
y_train = np.loadtxt("Dataset/train/y_train.txt")

X_test = np.loadtxt("Dataset/test/X_test.txt")
y_test = np.loadtxt("Dataset/test/y_test.txt")





def standardize(X_train, X_test):
    mu = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train_standardized = (X_train - mu)/std
    X_test_standardized = (X_test - mu)/std
    return X_train_standardized, X_test_standardized


X_train, X_test = standardize(X_train, X_test)








B_ = [1,5,10,20] # Bootstrap rounds

Ns_ = [10, 20, 50, 100, 200, 500, 1000, 2000] # Size of a bootstrap subset


def one_bootstrap(X_train, y_train, X_test, y_test, B, Ns):
    
    
    N_train = len(y_train)

    N_test = len(y_test)
    
    # Make a subset using SRS (without replacement)
    subset_indices = np.random.choice(N_train, Ns, replace=False)
    
    X_train_subset = X_train[subset_indices]
    y_train_subset = y_train[subset_indices]
    
    # Do the bootstrap
    y_test_pred = np.matrix(np.zeros((N_test,B)))
    
    for i in range(B):
        
        # Make bootstrap iid sample
        
        iid_indices = np.random.choice(Ns, Ns, replace=True)
        
        X_train_resample_subset = X_train_subset[iid_indices]
        y_train_resample_subset = y_train_subset[iid_indices]
        
        clf = LinearDiscriminantAnalysis(solver='svd')
        clf.fit(X_train_resample_subset, y_train_resample_subset)
        
        pred_i = clf.predict(X_test)
        y_test_pred[:,i] = pred_i.reshape(-1,1)
        
    # Do the majority vote
    y_test_majority = np.zeros((N_test)).astype(int)
    
    for i in range(N_test):
        y_pred_i = np.array(y_test_pred[i,:]).astype(int).reshape(-1)
        # print(y_pred_i)
        y_test_majority[i] = np.argmax(np.bincount(y_pred_i))    # return the most frequenced prediction
        
    return accuracy_score(y_test_majority, y_test)




## ============================
fig = plt.figure()  
ax = fig.add_subplot(1,1,1)

for B in B_:
    
    acc_i = -1
    
    accuracy = np.zeros((len(Ns_))) 
    
    for Ns in Ns_:
        
        acc_i = acc_i + 1 # index in accuracy score
        
        accuracy[acc_i] = one_bootstrap(X_train, y_train, X_test, y_test, B, Ns)
    
    plt.plot(Ns_, accuracy, label=str(B)+' Bootstraps')
    
ax.set_xscale('log') 
ax.set_xlabel("Sample size")
ax.set_ylabel("Accuracy")
ax.legend()
plt.show()