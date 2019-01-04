#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 1Advertising.py
# @time: 2018/12/13 上午10:18
# @desc:

import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

    plt.figure(facecolor='w')
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    plt.legend(loc='lower right')
    plt.xlabel(u'广告花费', fontsize=16, fontproperties=font_ch)
    plt.ylabel(u'销售额', fontsize=16, fontproperties=font_ch)
    plt.title(u'广告花费与销售额对比数据', fontsize=18, fontproperties=font_ch)
    plt.grid(b=True, ls=':')
    plt.show()

    plt.figure(facecolor='w', figsize=(9, 10))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid(b=True, ls=':')

    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid(b=True, ls=':')

    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid(b=True, ls=':')

    plt.tight_layout()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    linreg = LinearRegression()

    model = linreg.fit(x_train, y_train)
    print linreg.coef_
    print linreg.intercept_
    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]

    y_hat = linreg.predict(x_test)

    mse = np.array((y_hat - np.array(y_test)) ** 2)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print 'MSE = ', mse
    print 'RMSE = ', rmse
    print 'R2 = ', linreg.score(x_train, y_train)
    print 'R2 = ', linreg.score(x_test, y_test)

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper left')
    plt.title(u'线性回归预测销量', fontsize=18, fontproperties=font_ch)
    plt.grid(b=True, ls=':')
    plt.show()

