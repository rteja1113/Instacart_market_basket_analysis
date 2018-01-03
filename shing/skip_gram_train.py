#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:32:10 2017

@author: cvpr
"""
import numpy as np
import pandas as pd
import os
from Product2VecSkipGram2 import Product2VecSkipGram
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    path = '../../data/instacart/files'
    products = pd.read_csv(os.path.join(path, 'products.csv'))
    df = pd.read_pickle(os.path.join(path, 'prod2vec.pkl')).products
    print('initial size', len(df))

    df_train, df_cv = train_test_split(df, test_size=0.1, random_state=2017)
    batch_size = 1024
    rates = {100000: 0.5,
             200000: 0.25,
             500000: 0.1}
    model = Product2VecSkipGram(df_train, df_cv, batch_size, 1, 1, np.max(products.product_id) + 1)
    model.train(120001, 20000, len(df_cv) // batch_size, rates)