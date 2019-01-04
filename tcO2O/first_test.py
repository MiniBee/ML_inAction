#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: first_test.py
# @time: 2018/12/13 下午4:41
# @desc:

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt


# 数据清洗  有优惠券，为消费或15天内为消费，返回0；有优惠券，15天内有消费，返回1；其他返回-1
def get_label(data):
    if pd.isnull(data.Date) and not pd.isnull(data.Coupon_id):
        return 0
    elif not pd.isnull(data.Date) and not pd.isnull(data.Coupon_id):
        if (datetime.datetime.strptime(str(int(data.Date)),"%Y%m%d") - datetime.datetime.strptime(str(int(data.Date_received)),"%Y%m%d")).days <= 15:
            return 1
        else:
            return 0
    return -1


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


if __name__ == '__main__':
    train = pd.read_csv('./data/ccf_offline_stage1_train.csv')

    # train = train.head(100)

    test = pd.read_csv('./data/ccf_offline_stage1_test_revised.csv')
    train_set = pd.DataFrame()
    train_set['label'] = train.apply(get_label, axis=1)
    train_set['Distance'] = train.Distance.apply(lambda x: 0 if str(x) == 'nan' else x)
    train_set['rate'] = train['Discount_rate'].apply(get_discount_rate)
    train_set['type'] = train['Discount_rate'].apply(get_discount_type)
    train_set['man'] = train['Discount_rate'].apply(get_discount_man)
    train_set['jian'] = train['Discount_rate'].apply(get_discount_jian)
    train_set['weekday'] = train['Date_received'].apply(get_weekday)
    train_set['weekday_type'] = train_set['weekday'].apply(lambda x: 1 if x in [6,7] else 0)

    train_set = train_set[pd.notnull(train_set['weekday'])]

    weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
    temp = pd.get_dummies(train_set['weekday'].replace('nan', np.nan))
    temp.columns = weekdaycols
    train_set[weekdaycols] = temp

    model1 = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        shuffle=True,
        alpha=0.01,
        l1_ratio=0.01,
        n_jobs=1,
        class_weight=None
    )

    model = Pipeline([
        ('sc', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3)),
        ('sgdc', model1)
    ])

    original_feature = ['rate', 'type', 'man', 'jian', 'Distance', 'weekday', 'weekday_type'] + weekdaycols

    x_train, x_test, y_train, y_test = train_test_split(train_set[original_feature], train_set['label'], train_size=0.8, random_state=0)

    model.fit(x_train, y_train)

    test_set = pd.DataFrame()
    # test_set['label'] = test.apply(get_label, axis=1)
    test_set['Distance'] = test.Distance.apply(lambda x: 0 if str(x) == 'nan' else x)
    test_set['rate'] = test['Discount_rate'].apply(get_discount_rate)
    test_set['type'] = test['Discount_rate'].apply(get_discount_type)
    test_set['man'] = test['Discount_rate'].apply(get_discount_man)
    test_set['jian'] = test['Discount_rate'].apply(get_discount_jian)
    test_set['weekday'] = test['Date_received'].apply(get_weekday)
    test_set['weekday_type'] = test_set['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)

    weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
    temp = pd.get_dummies(test_set['weekday'].replace('nan', np.nan))
    temp.columns = weekdaycols
    test_set[weekdaycols] = temp

    test_set.to_csv('check.csv')

    test_predict = model.predict_proba(test_set[original_feature])
    # y_pred = model.predict(x_test)
    # test_fact = test_set['label']

    # print 'score', model.score(x_test, y_test)

    test_pre_new = test[['User_id', 'Coupon_id', 'Date_received']].copy()
    test_pre_new['label'] = test_predict[:, 1]
    test_pre_new.to_csv('submit1.csv', index=False, header=False)
    test_pre_new.head()











