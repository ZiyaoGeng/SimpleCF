"""
Created on Mar 18, 2020
Updated on Jan 25, 2021

Evaluation indicators

@author: Gengziyao(zggzy1996@163.com)
"""
import numpy as np
from tqdm import tqdm


def evaluate_model(model, test):
    """
    evaluate model
    :param model: model of CF
    :param test: dict.
    :return: hit rate, ndcg
    """
    hit, ndcg = 0, 0
    for user_id, item_id in tqdm(test.items()):
        item_rank = model.recommend(user_id)
        if item_id in item_rank:
            hit += 1
            ndcg += 1 / np.log2(item_rank.index(item_id) + 2)
    return hit / len(test), ndcg / len(test)