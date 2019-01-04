#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: kNN.py
# @time: 2018/12/1 下午5:50
# @desc:

import operator
import numpy as np


def create_dateset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables


def classify0(inx, dataSet, labels, k):
    dataset_size = dataSet.shape[0]



if __name__ == '__main__':
    group, lables = create_dateset()
    inx = [2.0, 2.0]
    classify0(inx, group, lables, 2)
