#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 3DecisionTreeRegpressor.py
# @time: 2018/12/15 下午8:03
# @desc:

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor


if __name__ == '__main__':
    N = 100
    x = np.random.rand(N) * 6 - 3  # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    print y
    x = x.reshape(-1, 1)
    print x

    dt = DecisionTreeRegressor(criterion='mse', max_depth=9)
    dt.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = dt.predict(x_test)

    plt.figure(facecolor='w')
    plt.plot(x, y, 'r*', markersize=10, markeredgecolor='k', label='fact')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label='predict')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(b=True, ls=':', color='#606060')
    plt.title('DecistionTreeRegressor', fontsize=15)
    plt.tight_layout(2)
    plt.show()





