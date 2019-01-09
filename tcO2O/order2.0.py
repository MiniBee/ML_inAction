#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: order2.0.py
# @time: 2018/12/14 下午2:14
# @desc:

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import lightgbm as lgb

import util
import datetime
import time

import matplotlib.pyplot as plt
import seaborn as sns


# https://tianchi.aliyun.com/notebook/detail.html?spm=5176.8366600.0.0.411a311f7kvfwx&id=4796
# https://tianchi.aliyun.com/notebook/detail.html?spm=5176.8366600.0.0.411a311f8jCgNp&id=23504
def procss_data(data):
    data['discount_rate'] = data['Discount_rate'].apply(util.get_discount_rate)
    data['discount_man'] = data['Discount_rate'].apply(util.get_discount_man)
    data['discount_jian'] = data['Discount_rate'].apply(util.get_discount_jian)
    data['discount_type'] = data['Discount_rate'].apply(util.get_discount_type)
    print(data['discount_rate'].unique())

    # convert distance
    data['Distance'] = data['Distance'].replace(np.nan, -1).replace('nan', -1).astype(int)
    return data


def check_date(train_pd):
    train_pd[['Date']] = train_pd[['Date']].astype(str)
    date_received = train_pd['Date_received'].unique()
    date_received = sorted(date_received[pd.notna(date_received)])
    date_buy = train_pd['Date'].unique()
    print date_buy
    date_buy = sorted(date_buy[pd.notna(date_buy)])
    date_buy = sorted(train_pd[pd.notna(train_pd['Date'])]['Date'])

    couponbydate = train_pd[pd.notna(train_pd['Date_received'])]\
        [['Date_received', 'Date']]
    couponbydate = couponbydate.groupby(['Date_received'], as_index=False).count()
    couponbydate.columns = ['Date_received', 'count']
    print couponbydate.count()

    buybydate = train_pd[(pd.notna(train_pd['Date'])) & (pd.notna(train_pd['Date_received']))][['Date_received', 'Date']]
    buybydate = buybydate.groupby(['Date_received'], as_index=False).count()
    buybydate.columns = ['Date_received', 'count']
    print buybydate.count()

    plt.figure(figsize=(12, 8))
    date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')

    sns.set_style('ticks')
    sns.set_context("notebook", font_scale=1.4)
    plt.subplot(211)
    plt.bar(date_received_dt, couponbydate['count'], label='number of coupon received')
    plt.bar(date_received_dt, buybydate['count'], label='number of coupon used')
    plt.yscale('log')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(212)
    plt.bar(date_received_dt, buybydate['count'] / couponbydate['count'])
    plt.ylabel('Ratio(coupon used/coupon received)')
    plt.tight_layout()
    plt.show()


def get_weedday(row):
    if row == 'nan' or pd.isna(row):
        return np.nan
    else:
        return datetime.date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


def label(data):
    if pd.isna(data.Date) and not pd.isna(data.Coupon_id) and str(data.Date) != 'nan':
        return 0
    elif not pd.isna(data.Date) and not pd.isna(data.Coupon_id) and str(data.Date) != 'nan':
        if (datetime.datetime.strptime(str(int(data.Date)), "%Y%m%d") - datetime.datetime.strptime(
                str(int(data.Date_received)), "%Y%m%d")).days <= 15:
            return 1
        else:
            return 0
    return -1


def check_model(data, predictors):
    data = data.fillna(0)
    classifier = lambda: SGDClassifier(
        loss='log',
        penalty='elasticnet',
        fit_intercept=True,
        max_iter=100,
        shuffle=True,
        n_jobs=1,
        class_weight=None)

    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]
    }

    classifier = lambda: lgb.LGBMClassifier(
        learning_rate=0.01,
        boosting_type='gbdt',
        objective='binary',
        metric='multi_logloss',
        max_depth=5,
        sub_feature=0.7,
        num_leaves=3,
        colsample_bytree=0.7,
        n_estimators=5000,
        verbose=-1)

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('en', classifier())
    ])

    print '-------------', data.groupby(['label']).count()

    model.fit(data[predictors], data['label'])

    # folder = StratifiedKFold(n_splits=3, shuffle=True)
    #
    # grid_search = GridSearchCV(model, cv=folder)
    # grid_search = grid_search.fit(data[predictors],
    #                               data['label'])

    return model


def user_feature(df):
    u = df[['User_id']].copy().drop_duplicates()

    print df.info()
    u1 = df[pd.notna(df['Date_received'])][['User_id']].copy()
    u1['u_coupon_count'] = 1
    u1 = u1.groupby('User_id', as_index=False).count()
    print u1.head()

    u2 = df[pd.notna(df['Date'])][['User_id']].copy()
    u2['u_buy_count'] = 1
    u2 = u2.groupby('User_id', as_index=False).count()
    print u2.head()

    u3 = df[(pd.notna(df['Date'])) & (pd.notna(df['Date_received']))][['User_id']].copy()
    u3['u_buy_with_coupon'] = 1
    u3 = u3.groupby('User_id', as_index=False).count()
    print u3.head()

    u4 = df[pd.notna(df['Date'])][['User_id', 'Merchant_id']].copy()
    u4.drop_duplicates(inplace=True)
    u4.groupby('User_id', as_index=False).count()
    u4.rename(columns={'Merchant_id': 'u_merchant_count'}, inplace=True)
    print u4.head()

    utmp = df[(pd.notna(df['Date'])) & (pd.notna(df['Date_received']))][['User_id', 'Distance']].copy()
    utmp.replace(-1, np.nan, inplace=True)
    u5 = utmp.groupby(['User_id'], as_index=False).min()
    u5.rename(columns={'Distance': 'u_min_distance'}, inplace=True)
    u6 = utmp.groupby(['User_id'], as_index=False).max()
    u6.rename(columns={'Distance': 'u_max_distance'}, inplace=True)
    u7 = utmp.groupby(['User_id'], as_index=False).mean()
    u7.rename(columns={'Distance': 'u_mean_distance'}, inplace=True)
    u8 = utmp.groupby(['User_id'], as_index=False).median()
    u8.rename(columns={'Distance': 'u_median_distance'}, inplace=True)
    print u5.head()
    print u6.head()
    print u7.head()
    print u8.head()

    print type(u)

    user_feature = pd.merge(u, u1, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u2, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u3, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u4, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u5, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u6, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u7, on='User_id', how='left')
    user_feature = pd.merge(user_feature, u8, on='User_id', how='left')

    user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
        'u_coupon_count'].astype('float')
    user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
        'u_buy_count'].astype('float')
    user_feature = user_feature.fillna(0)

    print(user_feature.columns.tolist())
    return user_feature


def merchant_feature(df):
    print df.info()
    m = df[['Merchant_id']].copy().drop_duplicates()

    m1 = df[pd.notna(df['Date_received'])][['Merchant_id']].copy()
    m1['m_coupon_count'] = 1
    m1 = m1.groupby('Merchant_id', as_index=False).count()
    print m1.head()

    m2 = df[pd.notna(df['Date'])][['Merchant_id']].copy()
    m2['m_sale_count'] = 1
    m2 = m2.groupby('Merchant_id', as_index=False).count()
    print m2.head()

    m3 = df[(pd.notna(df['Date'])) & (pd.notna(df['Date_received']))][['Merchant_id']].copy()
    m3['m_sale_with_coupon'] = 1
    m3 = m3.groupby(['Merchant_id'], as_index=False).count()
    print m3.head()

    mtmp = df[(pd.notna(df['Date'])) & (pd.notna(df['Date_received']))][['Merchant_id', 'Distance']].copy()
    mtmp.replace(-1, np.nan, inplace=True)
    m4 = mtmp.groupby(['Merchant_id'], as_index=False).min()
    m4.rename(columns={'Distance': 'm_min_distance'}, inplace=True)
    m5 = mtmp.groupby(['Merchant_id'], as_index=False).max()
    m5.rename(columns={'Distance': 'm_max_distance'}, inplace=True)
    m6 = mtmp.groupby(['Merchant_id'], as_index=False).mean()
    m6.rename(columns={'Distance': 'm_mean_distance'}, inplace=True)
    m7 = mtmp.groupby(['Merchant_id'], as_index=False).median()
    m7.rename(columns={'Distance': 'm_median_distance'}, inplace=True)

    merchant_feature = pd.merge(m, m1, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m2, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m3, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m4, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m5, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m6, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m7, on='Merchant_id', how='left')

    merchant_feature['m_coupon_use_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / merchant_feature[
        'm_coupon_count'].astype('float')
    merchant_feature['m_sale_with_coupon_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / \
                                                  merchant_feature['m_sale_count'].astype('float')
    merchant_feature = merchant_feature.fillna(0)

    print(merchant_feature.columns.tolist())
    return merchant_feature


def user_merchant_feature(df):
    print df.info()
    um = df[['User_id', 'Merchant_id']].copy().drop_duplicates()
    print um.head()

    um1 = df[['User_id', 'Merchant_id']].copy()
    um1['um_count'] = 1
    um1 = um1.groupby(['User_id', 'Merchant_id'], as_index=False).count()
    print um1.head()

    um2 = df[pd.notna(df['Date'])][['User_id', 'Merchant_id']].copy()
    um2['um_buy_count'] = 1
    um2 = um2.groupby(['User_id', 'Merchant_id'], as_index=False).count()
    print um2.head()

    um3 = df[pd.notna(df['Date_received'])][['User_id', 'Merchant_id']].copy()
    um3['um_coupon_count'] = 1
    um3 = um3.groupby(['User_id', 'Merchant_id'], as_index=False).count()
    print um3.head()

    um4 = df[(pd.notna(df['Date_received'])) & (pd.notna(df['Date']))][['User_id', 'Merchant_id']].copy()
    um4['um_buy_with_coupon'] = 1
    um4 = um4.groupby(['User_id', 'Merchant_id'], as_index=False).count()
    print um4.head()

    user_merchant_feature = pd.merge(um, um1, on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature = pd.merge(user_merchant_feature, um2, on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature = pd.merge(user_merchant_feature, um3, on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature = pd.merge(user_merchant_feature, um4, on=['User_id', 'Merchant_id'], how='left')
    user_merchant_feature = user_merchant_feature.fillna(0)

    user_merchant_feature['um_buy_rate'] = user_merchant_feature['um_buy_count'].astype('float') / \
                                           user_merchant_feature['um_count'].astype('float')
    user_merchant_feature['um_coupon_use_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float') / \
                                                  user_merchant_feature['um_coupon_count'].astype('float')
    user_merchant_feature['um_buy_with_coupon_rate'] = user_merchant_feature['um_buy_with_coupon'].astype('float') / \
                                                       user_merchant_feature['um_buy_count'].astype('float')
    user_merchant_feature = user_merchant_feature.fillna(0)

    print(user_merchant_feature.columns.tolist())
    return user_merchant_feature


def feature_process(feature, train, test):
    user_feature_ = user_feature(feature)
    merchant_feature_ = merchant_feature(feature)
    user_merchant_feature_ = user_merchant_feature(feature)
    print train.columns.tolist()
    print user_feature_.columns.tolist()
    train = pd.merge(train, user_feature_, on='User_id', how='left')
    train = pd.merge(train, merchant_feature_, on='Merchant_id', how='left')
    train = pd.merge(train, user_merchant_feature_, on=['User_id', 'Merchant_id'], how='left')
    train = train.fillna(0)

    test = pd.merge(test, user_feature_, on='User_id', how='left')
    test = pd.merge(test, merchant_feature_, on='Merchant_id', how='left')
    test = pd.merge(test, user_merchant_feature_, on=['User_id', 'Merchant_id'], how='left')
    test = test.fillna(0)

    return train, test


if __name__ == '__main__':
    start_time = time.time()
    pd.set_option('display.width', 1800)
    pd.set_option('display.max_columns', 40)
    train_pd = pd.read_csv('./data/ccf_offline_stage1_train.csv')
    # train_pd = pd.read_csv('./data/train_sample.csv')
    test_pd = pd.read_csv('./data/ccf_offline_stage1_test_revised.csv')

    # print train_pd.head()
    # print train_pd.info()

    # print train_pd['Discount_rate'].unique()
    # print train_pd['Distance'].unique()
    train_pd = procss_data(train_pd)
    test_pd = procss_data(test_pd)
    # check_date(train_pd)

    train_pd['weekday'] = train_pd['Date_received'].astype(str).apply(get_weedday)
    test_pd['weekday'] = test_pd['Date_received'].astype(str).apply(get_weedday)

    train_pd['weekday_type'] = train_pd['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)
    test_pd['weekday_type'] = test_pd['weekday'].apply(lambda x: 1 if x in [6, 7] else 0)

    weekdaycols = ['weekday_' + str(i) for i in range(1, 8)]
    tmpdf = pd.get_dummies(train_pd['weekday'].replace('nan', np.nan))
    tmpdf.columns = weekdaycols
    train_pd[weekdaycols] = tmpdf

    tmpdf = pd.get_dummies(test_pd['weekday'].replace('nan', np.nan))
    tmpdf.columns = weekdaycols
    test_pd[weekdaycols] = tmpdf

    train_pd['label'] = train_pd.apply(label, axis=1)
    # test_pd['label'] = test_pd.apply(label, axis=1)
    #
    original_feature1 = ['User_id', 'discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'Distance', 'weekday',
                        'label', 'weekday_type'] + weekdaycols

    original_feature = ['User_id', 'discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'Distance',
                         'weekday',
                         'weekday_type'] + weekdaycols
    # model = check_model(train_pd, original_feature)
    # print model.best_score_
    # print model.best_params_

    # user_feature(train_pd)
    # merchant_feature(train_pd)
    # user_merchant_feature(train_pd)
    train_pd, test_pd = feature_process(train_pd, train_pd, test_pd)
    print train_pd.columns.tolist()
    predictors = ['discount_rate', 'discount_man', 'discount_jian', 'discount_type', 'Distance',
                  'weekday', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
                  'weekday_7', 'weekday_type',
                  'u_coupon_count', 'u_buy_count', 'u_buy_with_coupon', 'u_merchant_count', 'u_min_distance',
                  'u_max_distance', 'u_mean_distance', 'u_median_distance', 'u_use_coupon_rate',
                  'u_buy_with_coupon_rate',
                  'm_coupon_count', 'm_sale_count', 'm_sale_with_coupon', 'm_min_distance',
                  'm_max_distance', 'm_mean_distance', 'm_median_distance', 'm_coupon_use_rate',
                  'm_sale_with_coupon_rate', 'um_count','um_buy_count',
                  'um_coupon_count', 'um_buy_with_coupon', 'um_buy_rate', 'um_coupon_use_rate',
                  'um_buy_with_coupon_rate']

    trainSub, validSub = train_test_split(train_pd, test_size=0.2, stratify=train_pd['label'], random_state=100)

    model = check_model(trainSub, predictors)
    # validSub['pred_prob'] = model.predict_proba(validSub[predictors])[:, 1]
    # print validSub[validSub['label']==1][['label', 'pred_prob']]
    # validgroup = validSub.groupby(['Coupon_id'])
    # aucs = []
    # for i in validgroup:
    #     tmpdf = i[1]
    #     if len(tmpdf['label'].unique()) != 2:
    #         continue
    #     fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    #     aucs.append(auc(fpr, tpr))
    # print 'aucs: ', np.average(aucs)
    y_test_pred = model.predict_proba(test_pd[predictors])
    submit = test_pd[['User_id', 'Coupon_id', 'Date_received']].copy()
    submit['label'] = y_test_pred[:, 1]
    submit.to_csv('submit2.csv', index=False, header=False)
    submit.head()
    print u'总耗时：', time.time() - start_time


