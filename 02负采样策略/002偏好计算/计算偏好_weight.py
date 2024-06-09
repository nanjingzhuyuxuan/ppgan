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
        data_interest = {i : {}for i in range(1,self.user_num+1)}
        max_item_impact = 0
        max_user_impact = 0
        for u in range(1,self.user_num+1):
            for i in range(1,self.item_num+1):
                item_impact = 1 / (np.log(1 + self.item_pop[i]))
                user_impact = self.user_pop[u]
                if max_item_impact <  item_impact:
                    max_item_impact = item_impact
                if max_user_impact <  user_impact:
                    max_user_impact = user_impact

        for u in range(1, self.user_num + 1):
            for i in range(1, self.item_num + 1):
                item_impact = (np.log(1 + self.item_pop[i]))
                # print(item_impact/max_item_impact,self.user_pop[u]/max_user_impact *  (self.item_pop[i]/self.max_item_pop))
                # w =  - self.config['r']*(1/(np.log(1+self.item_pop[i]))) + (self.user_pop[u]) *  (1-self.config['r'])
                # print(item_impact,self.user_pop[u] *  (self.item_pop[i]))
                # w =   item_impact + self.user_pop[u] *  self.item_pop[i]
                # w =  item_impact *  1/(np.abs( self.user_pop[u] -  self.item_pop[i]))   #  11
                w = item_impact * 1 / (np.abs(self.user_pop[u] - self.item_pop[i]))
                data_interest[u].update({i:w})


        for index ,item in self.train.iterrows():
            # data_interest[item['uid']][item['iid']] = 999
            try:
                data_interest[item['uid']].pop(item['iid'])
            except:
                continue
        self.res = {i : []for i in range(1,self.user_num+1)}
        for full in range(1, 11):
            print(full)
            for uid,item in data_interest.items():
                items_sort = sorted(item.items(),key=lambda x: x[1],reverse=False)[:int(len(item)*full/10)]

                for it in items_sort:
                    if it[1] == 999:
                        break
                    else:
                        self.res[uid].append(it[0])
            self.compute_error_rate()
            np.save(self.config['res_data']+'user_uninterest_item_weight'+'_'+str(full)+'.npy',self.res)
    def compute_error_rate(self):
        err = 0
        num = 1
        for i in range(1,self.user_num+1):
            if len(self.testSet[i])==0:
                continue
            hit = len(list(set(self.testSet[i]) & set(self.res[i])))
            err_item = hit / len(self.testSet[i])
            err += err_item
            num +=1
        err_res = err/num
        print("error   "+str(err_res))
if __name__ == '__main__':
    config = {
        '100k': {
            'split_gap': ',',
            'train_data': '../../01数据集/100k/weight_train_pop.csv',
            'test_data': '../../01数据集/100k/test.csv',
            'res_data': '../../01数据集/100k/',
        },
        'ml1m': {
            'split_gap': ',',
            'step_time': 17 * 24 * 60 * 60,
            'pop_time_gap': 30 * 24 * 60 * 60,
            'train_data': '../../01数据集/ml1m/weight_train_pop.csv',
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


    data_setname = '100k'
    data_config = config[data_setname]

    P = Pop(data_config)
    P.get_record_pop()
