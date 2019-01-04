#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: 0GDA.py
# @time: 2018/12/13 上午10:14
# @desc:

import math


if __name__ == '__main__':
    learning_rate = 0.01
    for a in range(1, 100):
        cur = 0
        for i in range(1000):
            cur -= learning_rate * (cur ** 2 - a)
        print u'%d的平方根为: %.8f, 真实值是: %.8f' % (a, cur, math.sqrt(a))

