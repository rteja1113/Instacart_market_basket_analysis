#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:22:43 2017

@author: cvpr
"""

import gc
import pandas as pd
import numpy as np
import os
import arboretum
import lightgbm as lgb
import json
import sklearn.metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.sparse import dok_matrix, coo_matrix
from sklearn.utils.multiclass import  type_of_target

def limit_df_mem(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype(np.int32)
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype(np.float32)
        else:
            continue
    return df

if __name__ == '__main__':
    path = '../../data/instacart/files'
    order_train_shing = pd.read_pickle(os.path.join(path, 'order_train_shing.pkl'))
    order_test_shing = pd.read_pickle(os.path.join(path, 'order_test_shing.pkl'))
    order_train_shing = limit_df_mem(order_train_shing)
    order_test_shing = limit_df_mem(order_test_shing)
    
    shing_features = [
        # 'reordered_dow_ration', 'reordered_dow', 'reordered_dow_size',
        # 'reordered_prev', 'add_to_cart_order_prev', 'order_dow_prev', 'order_hour_of_day_prev',
        'user_product_reordered_ratio', 'reordered_sum',
        'add_to_cart_order_inverted_mean', 'add_to_cart_order_relative_mean',
        'reorder_prob',
        'last', 'prev1', 'prev2', 'median', 'mean',
        'dep_reordered_ratio', 'aisle_reordered_ratio',
        'aisle_products',
        'aisle_reordered',
        'dep_products',
        'dep_reordered',
        'prod_users_unq', 'prod_users_unq_reordered',
        'order_number', 'prod_add_to_card_mean',
        'days_since_prior_order',
        'order_dow', 'order_hour_of_day',
        'reorder_ration',
        'user_orders', 'user_order_starts_at', 'user_mean_days_since_prior',
        # 'user_median_days_since_prior',
        'user_average_basket', 'user_distinct_products', 'user_reorder_ratio', 'user_total_products',
        'prod_orders', 'prod_reorders',
        'up_order_rate', 'up_orders_since_last_order', 'up_order_rate_since_first_order',
        'up_orders', 'up_first_order', 'up_last_order', 'up_mean_cart_position',
        # 'up_median_cart_position',
        'days_since_prior_order_mean',
        # 'days_since_prior_order_median',
        'order_dow_mean',
        # 'order_dow_median',
        'order_hour_of_day_mean',
        # 'order_hour_of_day_median'
    ]
    shing_features.extend(list(range(32)))
    
    my_features = ['days_since_ratio', 'product_reorder_rate_recent',
                   'UP_lifespan_vs_time_since_last', 'days_since_cumulative_ratio',
                   'days_since_prior_lag0_ratio', 'user_avg_product_diversity',
                   'days_since_prior_lag1_ratio', 'user_average_days_between_orders_recent',
                   'user_avg_reorder_size_recent', 'user_total_distinct_items_recent', 'UA_time_since_last_order', 
                   'UP_delta_hour_vs_last', 'user_reordered_lag0', 'user_std_days_between_orders']    
    
    order_train = pd.read_csv(os.path.join(path,'cleaned_train/valid_onr_0.csv'), usecols=['user_id', 'product_id']+my_features)
    order_test = pd.read_csv(os.path.join(path, 'cleaned_test/test.csv'), usecols=['user_id', 'product_id']+my_features)
    order_train = limit_df_mem(order_train)
    order_test  = limit_df_mem(order_test)
    
    
    order_train_shing = pd.merge(order_train_shing, order_train, on=['user_id', 'product_id'], how='left')
    order_test_shing = pd.merge(order_test_shing, order_test, on=['user_id', 'product_id'], how='left')
    del order_train, order_test
    gc.collect()
    categories = ['product_id', 'aisle_id', 'department_id']
    
    shing_features.extend(my_features)
    shing_features.extend(categories)
    cat_features =  ','.join([str(shing_features.index(cat)) for cat in categories])
    
    
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 256,
        'min_sum_hessian_in_leaf': 20,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        # 'bagging_fraction': 0.9,
        # 'bagging_freq': 3,
        'verbose': 1
    }

    data = order_train_shing[shing_features]
    labels = order_train_shing[['reordered']].values.astype(np.float32).flatten()
    data_val = order_test_shing[shing_features]
    
    assert data.shape[0] == 8474661
    
    lgb_train = lgb.Dataset(data, labels, categorical_feature=cat_features)

    evals_result = {}
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=450,
                    evals_result=evals_result)
                     
    prediction = gbm.predict(data_val)
    # prediction = model.predict(data_val)
    orders = order_test_shing.order_id.values
    products = order_test_shing.product_id.values

    result = pd.DataFrame({'product_id': products, 'order_id': orders, 'prediction': prediction})
    result.to_pickle(os.path.join(path, 'prediction_combine_features_v2.pkl'))
    