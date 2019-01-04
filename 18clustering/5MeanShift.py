#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 5MeanShift.py
# @time: 2018/12/21 上午9:50
# @desc:

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.cluster import MeanShift
from sklearn.metrics import euclidean_distances
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score,\
    adjusted_rand_score, silhouette_score

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


if __name__ == "__main__":
    N = 1000
    centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
    data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)

    m = euclidean_distances(data, squared=True)
    bw = np.median(m)
    print(bw)
    for i, mul in enumerate(np.linspace(0.1, 0.4, 8)):
        band_width = mul * bw
        model = MeanShift(bin_seeding=False, bandwidth=band_width)
        ms = model.fit(data)
        centers = ms.cluster_centers_
        y_pred = ms.labels_
        n_clusters = np.unique(y_pred).size
        print u'带宽：', mul, band_width, u'聚类簇的个数为：', n_clusters
        print 'Homogeneity：', homogeneity_score(y, y_pred)
        print 'completeness：', completeness_score(y, y_pred)
        print 'V measure：', v_measure_score(y, y_pred)
        print 'AMI：', adjusted_mutual_info_score(y, y_pred)
        print 'ARI：', adjusted_rand_score(y, y_pred)
        if n_clusters > 1:
            print 'Silhouette：', silhouette_score(data, y_pred), '\n'
        plt.scatter(data[:, 0], data[:, 1], c=y_pred, s=200, marker='.', edgecolors='k')
        plt.show()



