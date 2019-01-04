#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 5Boston.py
# @time: 2018/12/13 下午3:23
# @desc:

import sys

from sklearn.ensemble import RandomForestRegressor

reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error


def not_empty(s):
    return s != ''


if __name__ == '__main__':
    path = 'housing.data'
    file_data = pd.read_csv(path, header=None)
    data = np.empty((len(file_data), 14))
    for i, d in enumerate(file_data.values):
        d = list(map(float, list(filter(not_empty, d[0].split(' ')))))
        data[i] = d
    x, y = np.split(data, (13, ), axis=1)
    y = y.ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    model = Pipeline([
        ('sc', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=True)),
        ('linear', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
                                  fit_intercept=False, max_iter=1e3, cv=3))
    ])

    # model = RandomForestRegressor(n_estimators=50, criterion='mse')

    model.fit(x_train, y_train)

    order = y_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order, :]
    y_pred = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print 'R2:', r2
    print 'mse: ', mse

    t = np.arange(len(y_pred))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', lw=2, label='真实值')
    plt.plot(t, y_pred, 'b-', lw=2, label='估计值')
    plt.legend(loc='best')
    plt.title('波士顿房价预测', fontsize=18)
    plt.xlabel('样本编号', fontsize=15)
    plt.ylabel('房屋价格', fontsize=15)
    plt.grid()
    plt.show()


