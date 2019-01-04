#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 5Random_Forest.py
# @time: 2018/12/17 上午10:56
# @desc:

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    path = './data/iris.data'
    data = pd.read_csv(path, header=None)
    x = data[[0, 1, 2, 3]]
    y = LabelEncoder().fit_transform(data[4])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, oob_score=True)


