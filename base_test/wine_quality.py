#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: wine_quality.py
# @time: 2018/12/28 下午5:22
# @desc:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# https://tianchi.aliyun.com/notebook/detail.html?spm=5176.11510288.0.0.4ccdb7bdlPlTlr&id=4662


def over_view(data):
    print data.head()
    print data.info()
    print data.describe()


def show_(data):
    plt.style.use('ggplot')
    colnm = data.columns.tolist()
    fig = plt.figure(figsize=(10, 6))
    for i in range(len(colnm)):
        plt.subplot(2, 6, i+1)
        sns.boxplot(data[colnm[i]], orient="v", width=0.5)
        plt.ylabel(colnm[i], fontsize=12)
    plt.tight_layout()
    plt.show()
    fig = plt.figure(figsize=(10, 8))
    for i in range(len(colnm)):
        plt.subplot(4, 3, i+1)
        data[colnm[i]].hist(bins=100)
        plt.xlabel(colnm[i], fontsize=12)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def show_acidity(data):
    acidityFeat = ['fixed acidity', 'volatile acidity', 'citric acid',
                   'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
    plt.figure(figsize=(10, 4), dpi=500)
    for i in range(len(acidityFeat)):
        plt.subplot(2, 3, i+1)
        v = np.log10(np.clip(data[acidityFeat[i]].values, a_min=0.001, a_max=None))
        plt.hist(v, bins=60)
        plt.xlabel('log(' + acidityFeat[i] + ')', fontsize=12)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pd.set_option('display.width', 1800)
    pd.set_option('display.max_columns', 40)
    red_path = './data/winequality-red.csv'
    white_path = './data/winequality-white.csv'
    r_data = pd.read_csv(red_path, sep=';')
    # over_view(r_data)
    # show_(r_data)
    show_acidity(r_data)

