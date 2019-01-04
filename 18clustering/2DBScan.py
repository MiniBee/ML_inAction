#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 2DBScan.py
# @time: 2018/12/18 下午10:25
# @desc:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import sklearn.datasets as ds
from sklearn.pipeline import Pipeline

from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score,\
    adjusted_rand_score, silhouette_score


if __name__ == '__main__':
    # data = pd.read_csv('./data/iris.data', header=None)
    # x = data[[2, 3]]
    # y = data[4]
    # y = LabelEncoder().fit_transform(y)
    #
    # # x = MinMaxScaler().fit_transform(x)
    # # x = PCA().fit_transform(x)
    #
    # model = DBSCAN(eps=0.4, min_samples=9)
    # y_pred = model.fit_predict(x)
    #
    # print y
    # print y_pred
    #
    # print 'Homogeneity:', homogeneity_score(y, y_pred)
    # print 'completeness:', completeness_score(y, y_pred)
    # print 'V measure:', v_measure_score(y, y_pred)
    # print 'AMI:', adjusted_mutual_info_score(y, y_pred)
    # print 'ARI:', adjusted_rand_score(y, y_pred)
    # print 'Silhouette:', silhouette_score(x, y_pred)

    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)
    # 0.5

    # data, y = ds.make_circles(n_samples=N, noise=.05, shuffle=False, random_state=0, factor=0.3)
    # 0.2

    model = Pipeline([
        ('dbs', DBSCAN())
    ])

    for i in np.arange(0.1, 5, 0.1):
        for j in range(1, 10):
            model.set_params(dbs__eps=i, dbs__min_samples=j)

            y_pred = model.fit_predict(data)
            y_unique = np.unique(y_pred)
            n_clusters = y_unique.size - (1 if -1 in y_pred else 0)

            try:
                if silhouette_score(data, y_pred) > 0.5:
                    print 'eps: ', i
                    print 'MinPts: ', j
                    print 'n_clusters: ', n_clusters
                    print 'Silhouette:', silhouette_score(data, y_pred)
                    print 'Homogeneity:', homogeneity_score(y, y_pred)
                    print 'completeness:', completeness_score(y, y_pred)
                    print 'V measure:', v_measure_score(y, y_pred)
                    print 'AMI:', adjusted_mutual_info_score(y, y_pred)
                    print 'ARI:', adjusted_rand_score(y, y_pred)
                    print '\n---------------------------------------'
                    plt.scatter(data[:, 0], data[:, 1], c=y_pred, s=200, marker='.', edgecolors='k')
                    plt.show()
            except:
                break







