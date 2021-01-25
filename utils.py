"""
Created on Mar 18, 2020
Updated on Jan 25, 2021

load ml-1m

@author: Gengziyao(zggzy1996@163.com)
"""
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_ml_1m(file, topN, trans_score=2, sample_num=-1):
    """
    :param file: A string. dataset path.
    :param topN: A scalar. The number of recommender list
    :param trans_score: A scalar. Greater than it is 1, and less than it is 0.
    :param sample_num: A scalar. The number of data.
    :return: user_item_dict, item_user_dict, item_hot_list, test_data
    """
    print('========== Data Preprocess Start =============')
    names = ['user_id', 'item_id', 'score', 'Timestamp']
    if sample_num > 0:
        data_df = pd.read_csv(file, sep="::", engine='python', iterator=True, header=None,
                          names=names)
        data_df = data_df.get_chunk(sample_num)
    else:
        data_df = pd.read_csv(file, sep="::", engine='python', names=names)
    data_df = data_df[data_df.score >= trans_score]  # trans score
    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])  # sort
    
    test_data = pd.DataFrame()
    for user_id, df in tqdm(data_df.groupby('user_id')):
        # use last i️nteracted movie for each user
        test_data = pd.concat([test_data, df.iloc[-1:]], axis=0)
    data_df = data_df.drop(index=test_data.index)
    # user_item_dict
    user_item_df = data_df.groupby('user_id')[['item_id', 'score']].apply(lambda x: list(zip(x['item_id'], x['score'])))
    user_item_dict = dict(zip(user_item_df.index, user_item_df.values))
    # item_user_dict
    item_user_df = data_df.groupby('item_id')[['user_id', 'score']].apply(lambda x: list(zip(x['user_id'], x['score'])))
    item_user_dict = dict(zip(item_user_df.index, item_user_df.values))
    # hot_item_list
    item_hot_list = data_df['item_id'].value_counts().index[:topN].tolist()
    test_data = dict(zip(test_data['user_id'], test_data['item_id']))
    return user_item_dict, item_user_dict, item_hot_list, test_data
