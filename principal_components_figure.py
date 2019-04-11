import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import warnings
import os

# Load all the datasets
X_train = np.loadtxt("Dataset/train/X_train.txt")
y_train = np.loadtxt("Dataset/train/y_train.txt")

X_test = np.loadtxt("Dataset/test/X_test.txt")
y_test = np.loadtxt("Dataset/test/y_test.txt")


# Variables that might be needed later
p = X_train.shape[1]
N = X_train.shape[0]


# Create transformation function
def transform(M):
    transformer = LinearDiscriminantAnalysis().fit(X_train, y_train)
    return transformer.transform(M)

X = transform(X_test)

'''
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING
'''
labels = ['walking', 'walking upstairs', 'walking downstairs', 'sitting', 'standing', 'laying']
color = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


classes = np.unique(y_test).astype(int)

pairs = [(0,1), (0,2), (1,2)]

for pair in pairs:
    fig = plt.figure()
    for k in classes:
        
        indices = [i for i in range(0, len(y_test)) if y_test[i] == k]
         
        print(indices)

        plt.scatter(X[indices, pair[0]], X[indices, pair[1]], c=color[k-1], label=labels[k-1], s=5)
        plt.legend()
        plt.xlabel(r'$\nu_' + str(pair[0]) + '$')
        plt.ylabel(r'$\nu_' + str(pair[1]) + '$')
    
    filename = 'PC' + str(pair[0]) + 'PC' + str(pair[1]) + '.png'
    plt.savefig(filename)

'''
for k in classes:
    k = int(k)
    indices = [i for i in range(0, len(y_test)) if y_test[i] == k]
    print(indices) 
    label = labels[int(k-1)]

    plt.scatter(X[indices, 0], X[indices, 1], c=color[k-1], label=label)
    plt.legend()
    plt.savefig('PC0PC1.png')
    plt.scatter(X[indices, 0], X[indices, 2], c=color[k-1], label=label)
    plt.legend()
    plt.savefig('PC0PC2.png')
    plt.scatter(X[indices, 1], X[indices, 2], c=color[k-1], label=label)
    plt.legend()
    plt.savefig('PC1PC2.png')
'''

