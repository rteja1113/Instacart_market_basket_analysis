#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:27:56 2017

@author: cvpr
"""

import gc
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    path = '../../data/instacart/files'
    order_prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"), dtype={'order_id': np.uint32,
                                                                                      'product_id': np.uint16,
                                                                                      'add_to_cart_order':np.uint8,
                                                                                      'reordered': bool})
    
    orders = pd.read_csv(os.path.join(path, "orders.csv"), dtype={'order_id':np.uint32,
                                                                  'user_id': np.uint32,
                                                                  'eval_set': 'category',
                                                                  'order_number':np.uint8,
                                                                  'order_dow': np.uint8,
                                                                  'order_hour_of_day': np.uint8
                                                                  })
    print('loaded')
    
    orders = orders.loc[orders.eval_set=='prior', :]
    orders_user = orders[['order_id', 'user_id']]
    labels = pd.merge(order_prior, orders_user, on='order_id')
    labels = labels.loc[:, ['user_id', 'product_id']].drop_duplicates()
    print(labels)
    print('save')
    print(labels.shape)
    print(labels.columns)
    labels.to_pickle(os.path.join(path, 'previous_products.pkl'))
