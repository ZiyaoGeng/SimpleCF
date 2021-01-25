"""
Created on Jan 25, 2021

A simple start of UserCF

@author: Gengziyao(zggzy1996@163.com)
"""
import os
import pickle
from time import time

from UserCF import *
from evaluate import *
from utils import *


def main():
    # ========================= Hyper Parameters =======================
    file = '../dataset/ml-1m/ratings.dat'
    trans_score = 1
    sample_num = 10000  # sample_num = -1 can load all data

    sim_user_topK = 10
    topN = 100
    path = 'sim_matrix/u2u_sim_' + str(sample_num) + '_' + str(sim_user_topK) + '.pkl'
    # ========================== load dataset  ===========================
    user_item_dict, item_user_dict, item_hot_list, test_data = load_ml_1m(
        file, topN, trans_score=trans_score, sample_num=sample_num)
    # =========================== u2u_sim ===============================
    # if you store the similarity matrix, you can load it to model
    u2u_sim = None
    if os.path.exists(path):
        u2u_sim = pickle.load(open(path, 'rb'))
    print("================ Build Model =================")
    t1 = time()
    model = UserCF(user_item_dict, item_user_dict, item_hot_list, sim_user_topK, topN, u2u_sim)
    if u2u_sim is None:
        pickle.dump(model.u2u_sim, open(path, 'wb'))
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
