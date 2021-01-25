"""
Created on Mar 19, 2020
Updated on Jan 25, 2021

Item-based collaborative filtering

@author: Gengziyao(zggzy1996@163.com)
"""
import math
from collections import defaultdict
from tqdm import tqdm


class ItemCF:
    def __init__(self, user_item_dict, item_hot_list, sim_item_topK, topN, i2i_sim=None):
        """
        Item-based collaborative filtering
        :param user_item_dict: A dict. {user1: [(item1, score),...], user2: ...}
        :param item_hot_list: A list. The popular movies list.
        :param sim_item_topK: A scalar. Choose topK items for calculate.
        :param topN: A scalar. The number of recommender list.
        :param i2i_sim: dict. If None, the model should calculate similarity matrix.
        """
        self.user_item_dict = user_item_dict
        self.item_hot_list = item_hot_list
        self.sim_item_topK = sim_item_topK
        self.topN = topN
        self.i2i_sim = self.__get_item_sim() if i2i_sim is None else i2i_sim

    def __get_item_sim(self):
        """
        calculate item similarity weight matrix
        :return: i2i_sim
        """
        i2i_sim = dict()
        item_cnt = defaultdict(int)  # Count the number of visits to the item
        for user, items in tqdm(self.user_item_dict.items()):
            for i, score_i in items:
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for j, score_j in items:
                    if i == j:
                        continue
                    i2i_sim[i].setdefault(j, 0)
                    i2i_sim[i][j] += 1 / math.log(len(items) + 1)  # punish hot items
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])  # Cosine similarity
        return i2i_sim

    def recommend(self, user_id):
        """
        recommend one user
        :param user_id: user's ID
        :return:
        """
        item_rank = dict()
        user_hist_items = self.user_item_dict[user_id]
        for i, score_i in user_hist_items:
            for j, wij in sorted(self.i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:self.sim_item_topK]:
                if j in user_hist_items:
                    continue

                item_rank.setdefault(j, 0)
                item_rank[j] += 1 * wij

        if len(item_rank) < self.topN:
            for i, item in enumerate(self.item_hot_list):
                if item in item_rank:
                    continue
                item_rank[item] = - i - 1  # rank score < 0
                if len(item_rank) == self.topN:
                    break
        item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:self.topN]

        return [i for i, score in item_rank]

    def recommend_all(self, test):
        """
        recommend all user of test raw_data
        :return:
        """
        user_recall_items = defaultdict(dict)
        for user in tqdm(test.keys()):
            user_recall_items[user] = self.recommend(user)
        return user_recall_items