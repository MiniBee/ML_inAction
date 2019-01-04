# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import SpectralClustering
from sklearn.metrics import euclidean_distances


n_samples = 1000
X, y = ds.make_circles(n_samples=n_samples, noise=.05, shuffle=False, random_state=0, factor=0.3)

plt.scatter(X[:, 0], X[:, 1])
plt.show()
