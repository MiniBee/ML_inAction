#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 2LinearRegression_CV.py
# @time: 2018/12/13 上午11:00
# @desc:

import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.preprocessing import PolynomialFeatures

reload(sys)
sys.setdefaultencoding('utf-8')


if __name__ == '__main__':
    path = './Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    font_ch = mpl.font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

    mpl.rcParams['font.sans-serif'] = [font_ch]
    mpl.rcParams['axes.unicode_minus'] = False

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    x_train = PolynomialFeatures(degree=3).fit_transform(x_train)
    x_test = PolynomialFeatures(degree=3).fit_transform(x_test)
    linreg = Ridge()
    linreg = Lasso()
    alpha_can = np.logspace(-3, 2, 10)
    print 'alpha_can: ', alpha_can
    model = GridSearchCV(linreg, param_grid={'alpha': alpha_can}, cv=9)
    model.fit(x_train, y_train)
    print u'超参数： \n', model.best_params_

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = pd.DataFrame(x_test)
    x_test = x_test.values[order, :]
    y_hat = model.predict(x_test)
    # print(model.score(x_test, y_test))
    # mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    # rmse = np.sqrt(mse)  # Root Mean Squared Error
    # print(mse, rmse)

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'b-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.show()


