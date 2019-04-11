import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import warnings
import os

color = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# Generate data för class 1
n = 1000
theta = np.random.uniform(0,2*np.pi, size=n)
r1 = np.random.uniform(0,1, n)

plt.scatter(r1*np.cos(theta), r1*np.sin(theta), c=color[0], s=5)

# Generate data for class 2

r2 = np.random.uniform(0,1,n) + 1.2
theta = np.random.uniform(0,2*np.pi, n)
plt.scatter(r2*np.cos(theta), r2*np.sin(theta), c=color[1], s=5)

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.axis('equal')

plt.savefig('figures/bad_data.png')

fig = plt.figure()
plt.scatter(r1, theta, c=color[0], s=5)
plt.scatter(r2, theta, c=color[1], s=5)

plt.xlabel(r'$r$')
plt.ylabel(r'$\theta$')

plt.savefig('figures/bad_data_transformed.png')
