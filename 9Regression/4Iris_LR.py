#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 4Iris_LR.py
# @time: 2018/12/13 下午2:49
# @desc:

import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


if __name__ == '__main__':
    path = './iris.data'

    def iris_type(s):
        it = {b'Iris-setosa': 0,
              b'Iris-versicolor': 1,
              b'Iris-virginica': 2}
        return it[s]

    data = pd.read_csv(path, header=None)
    data[4] = pd.Categorical(data[4]).codes
    x, y = np.split(data.values, (4,), axis=1)
    x = x[:, :2]

    lr = Pipeline([
        ('sc', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('clf', LogisticRegression())
    ])
    for t in range(1, 5):
        lr.set_params(poly__degree=t)
        lr.fit(x, y.ravel())
        y_hat = lr.predict(x)
        y_hat_prob = lr.predict_proba(x)

        np.set_printoptions(suppress=True)

        print 'degree = ', t
        # print 'y_hat = \n', y_hat
        # print 'y_hat_prob = \n', y_hat_prob
        print '准确度：%.2f%%' % (100*np.mean(y_hat == y.ravel()))


