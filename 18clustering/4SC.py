# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import SpectralClustering
from sklearn.metrics import euclidean_distances
import sklearn.datasets as ds
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score,\
    adjusted_rand_score, silhouette_score

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    # data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)

    data, y = ds.make_circles(n_samples=N, noise=.05, shuffle=False, random_state=0, factor=0.3)


    n_clusters = 2
    m = euclidean_distances(data, squared=True)

    for i, s in enumerate(np.logspace(-2, 0, 6)):
        af = np.exp(-m ** 2 / (s ** 2)) + 1e-6
        model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=1)
        y_pred = model.fit_predict(af)

        try:
            silhouette = silhouette_score(data, y_pred)
            print 'n_clusters: ', n_clusters
            print 'Silhouette:', silhouette_score(data, y_pred)
            print 'eps: ', i
            print 'Homogeneity:', homogeneity_score(y, y_pred)
            print 'completeness:', completeness_score(y, y_pred)
            print 'V measure:', v_measure_score(y, y_pred)
            print 'AMI:', adjusted_mutual_info_score(y, y_pred)
            print 'ARI:', adjusted_rand_score(y, y_pred)
        except:
            pass
        if silhouette > 0.2:
            print s
            plt.scatter(data[:, 0], data[:, 1], c=y_pred, s=100, marker='.', edgecolors='k')
            plt.show()
