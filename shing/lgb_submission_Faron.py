import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os
from datetime import datetime

class F1Optimizer():
    def __init__(self):
        pass

    @staticmethod
    def get_expectations(P, pNone=None):
        expectations = []
        P = np.sort(P)[::-1]

        n = np.array(P).shape[0]
        DP_C = np.zeros((n + 2, n + 1))
        if pNone is None:
            pNone = (1.0 - P).prod()

        DP_C[0][0] = 1.0
        for j in range(1, n):
            DP_C[0][j] = (1.0 - P[j - 1]) * DP_C[0, j - 1]

        for i in range(1, n + 1):
            DP_C[i, i] = DP_C[i - 1, i - 1] * P[i - 1]
            for j in range(i + 1, n + 1):
                DP_C[i, j] = P[j - 1] * DP_C[i - 1, j - 1] + (1.0 - P[j - 1]) * DP_C[i, j - 1]

        DP_S = np.zeros((2 * n + 1,))
        DP_SNone = np.zeros((2 * n + 1,))
        for i in range(1, 2 * n + 1):
            DP_S[i] = 1. / (1. * i)
            DP_SNone[i] = 1. / (1. * i + 1)
        for k in range(n + 1)[::-1]:
            f1 = 0
            f1None = 0
            for k1 in range(n + 1):
                f1 += 2 * k1 * DP_C[k1][k] * DP_S[k + k1]
                f1None += 2 * k1 * DP_C[k1][k] * DP_SNone[k + k1]
            for i in range(1, 2 * k - 1):
                DP_S[i] = (1 - P[k - 1]) * DP_S[i] + P[k - 1] * DP_S[i + 1]
                DP_SNone[i] = (1 - P[k - 1]) * DP_SNone[i] + P[k - 1] * DP_SNone[i + 1]
            expectations.append([f1None + 2 * pNone / (2 + k), f1])

        return np.array(expectations[::-1]).T

    @staticmethod
    def maximize_expectation(P, pNone=None):
        expectations = F1Optimizer.get_expectations(P, pNone)

        ix_max = np.unravel_index(expectations.argmax(), expectations.shape)
        max_f1 = expectations[ix_max]

        predNone = True if ix_max[0] == 0 else False
        best_k = ix_max[1]

        return best_k, predNone, max_f1

    @staticmethod
    def _F1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn)

    @staticmethod
    def _Fbeta(tp, fp, fn, beta=1.0):
        beta_squared = beta ** 2
        return (1.0 + beta_squared) * tp / ((1.0 + beta_squared) * tp + fp + beta_squared * fn)

import multiprocessing


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    df_list, max_f1 = zip(*retLst)
    return pd.concat(df_list), max_f1

def create_products_faron(df):
    # print(df.product_id.values.shape)
    products = df.product_id.values
    prob = df.prediction.values

    sort_index = np.argsort(prob)[::-1]
    L2 = products[sort_index]
    P2 = prob[sort_index]

    opt = F1Optimizer.maximize_expectation(P2)

    best_prediction = ['None'] if opt[1] else []
    best_prediction += list(L2[:opt[0]])

    #print("Prediction {} ({} elements) yields best E[F1] of {}\n".format(best_prediction, len(best_prediction), opt[2]))
    #print('iteration', df.shape[0], 'optimal value', opt[0])

    best = ' '.join(map(lambda x: str(x), best_prediction))
    df = df[0:1]
    df.loc[:, 'products'] = best
    return df, opt[2]

if __name__ == '__main__':
    path = '../../data/instacart/files'
    data_arb_v1 = pd.read_pickle(os.path.join(path, 'prediction_arboretum.pkl'))
    data_lgb = pd.read_pickle(os.path.join(path, 'prediction_combine_features.pkl'))
    data_lgb_v2 = pd.read_pickle(os.path.join(path, 'prediction_combine_features_v2.pkl'))
    data_arb_v2 = pd.read_pickle(os.path.join(path, 'prediction_arboretum_v2.pkl'))
    data_arb_v3 = pd.read_pickle(os.path.join(path, 'prediction_arboretum_v3.pkl'))
    data_arb_v4 = pd.read_pickle(os.path.join(path, 'prediction_arboretum_v4.pkl'))
    #df = data[['order_id', 'prediction', 'product_id']]
            'order_id':data_arb_v1.order_id,
            'prediction':(data_arb_v1.prediction+
                          data_lgb.prediction+
                          data_lgb_v2.prediction+
                          data_arb_v2.prediction+
                          data_arb_v3.prediction+
                          data_arb_v4.prediction)/6,
            'product_id':data_arb_v1.product_id})
    # Group products per order here
    
    df_order, max_f1 = applyParallel(df.groupby(df.order_id), create_products_faron)#.reset_index()
    
    #df_arb, max_f1_arb = applyParallel(data_arb.groupby(data_arb.order_id), create_products_faron)#.reset_index()
    #df_lgb, max_f1_lgb = applyParallel(data_lgb.groupby(data_lgb.order_id), create_products_faron)#.reset_index()
    

    df_order[['order_id', 'products']].to_csv(os.path.join(path, 'arb4_lgb3_Faron_Final.csv'), index=False)
    
    
    df_compare = pd.DataFrame({
            'order_id':df_arb.order_id,
            'arb_products':df_arb.products,
            'lgb_products':df_lgb.products})
    
    f1_compare = np.array([list(max_f1_arb), list(max_f1_lgb)])
    f1_compare = f1_compare.T
    df_compare['which_max'] = f1_compare.argmax(axis=1)

    for i,row in df_compare.iterrows():
        if row.which_max:
            df_compare.loc[i, 'products'] = df_compare.loc[i, 'lgb_products']
        else:
            df_compare.loc[i, 'products'] = df_compare.loc[i, 'arb_products']
    
    df_compare[['order_id', 'products']].to_csv(os.path.join(path, 'arboretum_lgb_compareF1_Faron.csv'), index=False)