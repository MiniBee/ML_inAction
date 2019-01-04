#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 2DBScan.py
# @time: 2018/12/18 下午10:25
# @desc:

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score,\
    adjusted_rand_score, silhouette_score


if __name__ == '__main__':
    data = pd.read_csv('./data/iris.data', header=None)
    x = data[[2, 3]]
    y = data[4]
    y = LabelEncoder().fit_transform(y)

    x = MinMaxScaler().fit_transform(x)

    model = Pipeline([
        ('dbs', DBSCAN())
    ])

    for i in np.arange(0.1, 5, 0.1):
        for j in range(0, 100):
            model.set_params(dbs__eps=i, dbs__min_samples=j)

            y_pred = model.fit_predict(x)
            y_unique = np.unique(y_pred)
            n_clusters = y_unique.size - (1 if -1 in y_pred else 0)
            if n_clusters == 3:
                print np.unique(y_pred)
                print '\n---------------------------------------'
                print y_pred
                print i, j
                print 'Homogeneity:', homogeneity_score(y, y_pred)
                print 'completeness:', completeness_score(y, y_pred)
                print 'V measure:', v_measure_score(y, y_pred)
                print 'AMI:', adjusted_mutual_info_score(y, y_pred)
                print 'ARI:', adjusted_rand_score(y, y_pred)
                try:
                    print 'Silhouette:', silhouette_score(x, y_pred)
                except:
                    break




