"""
Created on Mar 18, 2020
Updated on Jan 25, 2021

User-based collaborative filtering

@author: Gengziyao(zggzy1996@163.com)
"""
import math
from collections import defaultdict
from tqdm import tqdm


class UserCF:
    def __init__(self, user_item_dict, item_user_dict, item_hot_list, sim_user_topK, topN, u2u_sim=None):
        """
        User-based collaborative filtering
        :param user_item_dict: A dict. {user1: [(item1, score),...], user2: ...}
        :param item_user_dict: A dict. {item1: [(user1, score),...], item2: ...}
        :param item_hot_list: A list. The popular movies list.
        :param sim_item_topK: A scalar. Choose topK items for calculate.
        :param topN: A scalar. The number of recommender list.
        :param i2i_sim: dict. If None, the model should calculate similarity matrix.
        """
        self.user_item_dict = user_item_dict
        self.item_user_dict = item_user_dict
        self.item_hot_list = item_hot_list
        self.sim_user_topK = sim_user_topK
        self.topN = topN
        self.u2u_sim = self.__get_user_sim() if u2u_sim is None else u2u_sim

    def __get_user_sim(self):
        """
        calculate user similarity weight matrix
        :return: u2u_sim
        """
        u2u_sim = dict()
        user_cnt = defaultdict(int)  # Count the number of visits to the user
        for item, users in tqdm(self.item_user_dict.items()):
            for i, score_i in users:
                user_cnt[i] += 1
                u2u_sim.setdefault(i, {})
                for j, score_j in users:
                    if i == j:
                        continue
                    u2u_sim[i].setdefault(j, 0)
                    u2u_sim[i][j] += 1 / math.log(len(users) + 1)  # punish highly active users
        for i, related_users in u2u_sim.items():
            for j, wij in related_users.items():
                u2u_sim[i][j] = wij / math.sqrt(user_cnt[i] * user_cnt[j])  # Cosine similarity
        return u2u_sim

    def recommend(self, user_id):
        """
        recommend one user
        :param user_id: user's ID
        :return:
        """
        item_rank = dict()
        user_hist_items = self.user_item_dict[user_id]
        for i, wij in sorted(self.u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:self.sim_user_topK]:
            for j, score_j in self.user_item_dict[i]:
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