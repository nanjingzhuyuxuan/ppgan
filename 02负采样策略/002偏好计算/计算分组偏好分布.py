'''
2022-12-16
当前版本只记录项目流行度，前后流行度
项目流行度的计算
（1）当前时间点流行度：基于当前时间点前后一段时间，计算该项目与所有项目的占比，（每个记录都应该有一个流行度）
（2）整个项目i的流行度：计算项目i的流行度均值，标准差

顺序:
先计算每条记录的流行度比例
计算项目的均值  标准差
'''
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from collections import defaultdict
import math

class Pop:
    def __init__(self,data_config):
        self.config = data_config

        self.train = []                     #uid  iid  rate  time  pop_value  is_pop
        self.test = []                      #uid  iid  rate  time  pop_value  is_pop

        self.item_num = 0
        self.user_num = 0

        self.get_data()


    def get_data(self):
        print('-----------加载数据---------')
        self.train = pd.read_csv(self.config['train_data'],names=['uid','iid','rate','time'],sep= self.config['split_gap'])
        self.train['rate'] = self.train['rate']/self.train['rate'].max()
        self.test = pd.read_csv(self.config['test_data'], names=['uid','iid','rate','time'],sep='\t')


        self.user_num = int(  max(self.train[['uid']].max()['uid'],self.test[['uid']].max()['uid']))
        self.item_num = int( max(self.train[['iid']].max()['iid'],self.test[['iid']].max()['iid']))

        self.testSet = {i:[] for i in range(1,self.user_num+1)}
        for index,item in self.test.iterrows():
            self.testSet[item['uid']].append(item['iid'])

        self.user_item_dict = {i:[] for i in range(1,self.user_num+1)}
        self.item_user_dict = {i:[] for i in range(1,self.item_num+1)}

        # 用户  项目流行度
        self.user_pop = {}

        self.item_pop = {}
        self.max_item_pop = 0
        for index,item in self.train.iterrows():
            self.user_item_dict[item['uid']].append(item['rate'])
            self.item_user_dict[item['iid']].append(item['rate'])

        for uid,pop_list in self.user_item_dict.items():

            self.user_pop.update({uid:sum(pop_list)/len(pop_list)})
        for iid,pop_list in self.item_user_dict.items():
            if self.max_item_pop < sum(pop_list)/len(pop_list):
                self.max_item_pop = sum(pop_list)/len(pop_list)
            self.item_pop.update({iid:sum(pop_list)/len(pop_list)})

    def get_record_pop(self):
        '''
        计算该项目时间点前后一段时间的流行度：该项目数/总项目数
        '''
        data_interest = {i : []for i in range(1,self.user_num+1)}
        # data_interest = []

        user_pop_sort = sorted(self.user_pop.items(),key=lambda x:x[1],reverse=True)

        for u in range(1, self.user_num + 1):
            for i in range(1, self.item_num + 1):
                item_impact = (np.log( 1 + self.item_pop[i]))
                w = item_impact*  1/(np.abs(self.user_pop[u] - self.item_pop[i]))
                # w = item_impact
                data_interest[u].append(w)
        u_pop_list = {}
        for u,set in data_interest.items():
            u_pop = np.mean(set)
            u_pop_list.update({u:u_pop})
        print(u_pop_list)
        data_u_set  = {i:[]for i in range(0,10)}
        for u in range(1, self.user_num + 1):
            u_pop = u_pop_list[u]//3
            if u_pop>9:

                data_u_set[9].append(u_pop_list[u])
            else:
                data_u_set[u_pop].append(u_pop_list[u])
        print(data_u_set)
        mean_list = []
        std_list = []
        for i,set in data_u_set.items():
            mean_u_pop = np.mean(set)
            std = np.var(set)
            mean_list.append([i,mean_u_pop])
            std_list.append([i,std])
        print(mean_list)
        print(std_list)


if __name__ == '__main__':
    config = {
        '100k': {
            'split_gap': ',',
            'full':0.7,
            'train_data': '../../01数据集/100k/train_pop.csv',
            'test_data': '../../01数据集/100k/test.csv',
            'res_data': '../../01数据集/100k/',
        },
        'ml1m': {
            'split_gap': ',',
            'step_time': 17 * 24 * 60 * 60,
            'pop_time_gap': 30 * 24 * 60 * 60,
            'train_data': '../../01数据集/ml1m/all_train_pop.csv',
            'test_data': '../../01数据集/ml1m/test.csv',
            'res_data': '../../01数据集/ml1m/',
        },
        'cd_movies': {
            'step_time': 60 * 24 * 60 * 60,
            'pop_time_gap': 180 * 24 * 60 * 60,
            'split_gap': ',',
            'train_data': '../../01数据集/cd_movies/train_pop.csv',
            'test_data': '../../01数据集/cd_movies/test.csv',
            'res_data': '../../01数据集/cd_movies/',
        },
    }


    data_setname = 'cd_movies'
    data_config = config[data_setname]

    P = Pop(data_config)
    P.get_record_pop()
