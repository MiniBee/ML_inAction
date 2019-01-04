#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 1kmeans.py
# @time: 2018/12/18 下午10:07
# @desc:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
import sklearn.datasets as ds

from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score,\
    adjusted_rand_score, silhouette_score


if __name__ == '__main__':
    # data = pd.read_csv('./data/iris.data', header=None)
    # x = data[[0, 1, 2, 3]]
    # y = data[4]
    # y = LabelEncoder().fit_transform(y)
    #
    # # x = MinMaxScaler().fit_transform(x)
    # # x = PCA().fit_transform(x)
    #
    # model = KMeans(n_clusters=3, n_init=10, max_iter=1000)
    # y_pred = model.fit_predict(x)
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

    # plt.scatter(data[:, 0], data[:, 1], c='b', s=200, marker='.', edgecolors='k')
    # plt.show()
    # exit(0)

    data, y = ds.make_circles(n_samples=N, noise=.05, shuffle=False, random_state=0, factor=0.3)

    print data

    sse_list = []

    for i in range(2, 10):
        model = KMeans(n_clusters=i, init='k-means++', n_init=5)
        y_pred = model.fit_predict(data)

        sse_list.append([i, model.inertia_])

        plt.scatter(data[:, 0], data[:, 1], c=y_pred, s=200, marker='.', edgecolors='k')

        print u'K：', i
        print 'SSE: ', model.inertia_
        print 'Homogeneity：', homogeneity_score(y, y_pred)
        print 'completeness：', completeness_score(y, y_pred)
        print 'V measure：', v_measure_score(y, y_pred)
        print 'AMI：', adjusted_mutual_info_score(y, y_pred)
        print 'ARI：', adjusted_rand_score(y, y_pred)
        print 'Silhouette：', silhouette_score(data, y_pred), '\n'
        plt.show()

    x = [i[0] for i in sse_list]
    y = [i[1] for i in sse_list]
    plt.plot(x, y, 'k-*')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.show()







