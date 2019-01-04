#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: util.py
# @time: 2018/12/14 下午4:14
# @desc:

import pandas as pd
import datetime
import numpy as np


# 折扣类型

# 1， 满减；0，打折
def get_discount_type(row):
    if not isinstance(row, unicode):
        row = str(row)
    if pd.isnull(row) or row == 'nan':
        return -1
    elif ':' in row:
        return 1
    else:
        return 0


def get_discount_rate(row):
    if not isinstance(row, unicode):
        row = str(row)
    if pd.isnull(row) or row == 'nan':
        return 1
    elif ':' in row:
        rows = row.split(':')
        return 1 - float(float(rows[1])/float(rows[0]))
    else:
        return float(row)


def get_discount_man(row):
    if not isinstance(row, unicode):
        row = str(row)
    if ':' in row:
        return int(row.split(':')[0])
    else:
        return 0


def get_discount_jian(row):
    if not isinstance(row, unicode):
        row = str(row)
    if ':' in row:
        return int(row.split(':')[1])
    else:
        return 0


def get_weekday(row):
    row = str(row)
    if row == 'nan':
        return np.nan
    else:
        return datetime.date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

