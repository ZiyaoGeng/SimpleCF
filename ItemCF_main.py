"""
Created on Jan 25, 2021

A simple start of ItemCF

@author: Gengziyao(zggzy1996@163.com)
"""
import os
import pickle
from time import time

from ItemCF import *
from evaluate import *
from utils import *


def main():
    # ========================= Hyper Parameters =======================
    file = '../dataset/ml-1m/ratings.dat'
    trans_score = 1
    sample_num = 10000  # sample_num = -1 can load all data

    sim_item_topK = 10
    topN = 100
    path = 'sim_matrix/i2i_sim_' + str(sample_num) + '_' + str(sim_item_topK) + '.pkl'
    # ========================== load dataset  ===========================
    user_item_dict, _, item_hot_list, test_data = load_ml_1m(file, topN, trans_score=trans_score, sample_num=sample_num)
    # =========================== i2i_sim ===============================
    # if you store the similarity matrix, you can load it to model
    i2i_sim = None
    if os.path.exists(path):
        i2i_sim = pickle.load(open(path, 'rb'))
    print("================ Build Model =================")
    t1 = time()
    model = ItemCF(user_item_dict, item_hot_list, sim_item_topK, topN, i2i_sim)
    if i2i_sim is None:
        pickle.dump(model.i2i_sim, open(path, 'wb'))
    t2 = time()
    # =========================== recommend ===============================
    # item_rank = model.recommend(1)
    print("================== Evaluate ==================")
    hr, ndcg = evaluate_model(model, test_data)
    t3 = time()
    print('Calculate similarity matrix [%d s], Evaluate[%d s]: HR = %f, NDCG = %f'
          % (t2 - t1, t3 - t2, hr, ndcg))


if __name__ == '__main__':
    main()
