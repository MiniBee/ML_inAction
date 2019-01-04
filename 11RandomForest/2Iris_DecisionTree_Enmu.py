#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 2Iris_DecisionTree_Enmu.py
# @time: 2018/12/15 下午7:54
# @desc:

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    path = './data/iris.data'
    data = pd.read_csv(path, header=None)

    y = LabelEncoder().fit(data[4])
    x = data[[0, 1, 2, 3]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    model.predict(x_test)




