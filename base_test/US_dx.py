#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: hongyue.pei
# @file: US_dx.py
# @time: 2018/12/28 下午2:32
# @desc:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}


occupation_map = {
  'INFORMATION REQUESTED PER BEST EFFORTS':'NOT PROVIDED',
  'INFORMATION REQUESTED':'NOT PROVIDED',
  'SELF' : 'SELF-EMPLOYED',
  'SELF EMPLOYED' : 'SELF-EMPLOYED',
  'C.E.O.':'CEO',
  'LAWYER':'ATTORNEY',
}


def over_view(data):
    print data.head()
    print data.info()
    print data.describe()


def fill_na(data):
    data['contbr_employer'].fillna('NOT PROVIDED', inplace=True)
    data['contbr_occupation'].fillna('NOT PROVIDED', inplace=True)


def amt_groupby_occupation(data):
    # print data.groupby('contbr_occupation')['contb_receipt_amt'].sum().sort_values(ascending=False)[:30]
    data.contbr_occupation = data.contbr_occupation.map(lambda x: occupation_map.get(x, x))


def cut_amt(data):
    bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
    labels = pd.cut(data['contb_receipt_amt'], bins)
    # print labels


def pivot_table(data):
    by_occupation = data.pivot_table('contb_receipt_amt', index='contbr_occupation', columns='party', aggfunc='sum')
    # print by_occupation.head(100)
    # by_occupation.plot(kind='bar')
    # plt.show()


if __name__ == '__main__':
    pd.set_option('display.width', 1800)
    pd.set_option('display.max_columns', 40)
    file_path = './data/data_03.csv'
    data = pd.read_csv(file_path)
    # print 'before fill na'
    # over_view(data)
    fill_na(data)
    # print 'after fill na'
    # over_view(data)
    # 添加党派
    data['party'] = data['cand_nm'].map(parties)
    # over_view(data)
    amt_groupby_occupation(data)
    # print data.groupby('contbr_occupation')['contb_receipt_amt'].sum().sort_values(ascending=False)[:30]
    data = data[data['contb_receipt_amt']>0]
    # print data.groupby('contbr_occupation')['contb_receipt_amt'].sum().sort_values(ascending=False)[:30]
    # data_vs = data[data['cand_nm'].isin(['Obama, Barack', 'Romney, Mitt'])].copy()
    print data.info()
    cut_amt(data)
    pivot_table(data)




