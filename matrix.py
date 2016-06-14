#encoding=utf8

import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform, cdist

x = np.random.normal(size=(1000, 1000))

K = np.zeros(1000)

for i in range(1000):
    K[i] = np.exp(-1 * (np.linalg.norm(x[i] - x[0]) ** 1))

print K

print np.exp(-1 * (cdist(x, np.atleast_2d(x[0]), 'euclidean') ** 1)).T

'''
print x

K = np.zeros((1000, 1000))

for i in range(1000):
    for j in range(1000):
        K[i, j] = (100 + 2 * np.dot(x[i], x[j])) ** 2

print K

print (100 + 2 * np.dot(x, x.T)) ** 2

K = np.zeros((1000, 1000))

for i in range(1000):
    for j in range(1000):
        K[i, j] = np.exp(-1 * (np.linalg.norm(x[i] - x[j]) ** 1))

print K

K = np.zeros((1000, 1000))

pairwise_dists = squareform(pdist(x, 'euclidean'))
K = np.exp(-1 * (pairwise_dists ** 1))

print K
'''

