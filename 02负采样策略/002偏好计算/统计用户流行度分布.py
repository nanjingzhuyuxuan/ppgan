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
import seaborn as sns
import pandas as pd
import json
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt


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
            self.user_item_dict[item['uid']].append(round(item['rate'],3))
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
        data  = []
        for u in range(1, self.user_num + 1):
            for r in self.user_item_dict[u]:
                if r<0:
                    continue
                else:
                    data.append([round(self.user_pop[u],4),r])


        df = pd.DataFrame(data,columns=['x','y'])

        plt.scatter(df.x, df.y, color='#023047',  alpha=0.3,s=5)
        plt.xlabel('user_pop')
        plt.ylabel('item_pop')

        plt.show()
if __name__ == '__main__':
    config = {
        '100k': {
            'split_gap': ',',
            'full':0.7,
            'train_data': '../../01数据集/100k/weight_train_pop.csv',
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
            'train_data': '../../01数据集/cd_movies/front_train_pop.csv',
            'test_data': '../../01数据集/cd_movies/test.csv',
            'res_data': '../../01数据集/cd_movies/',
        },
    }


    data_setname = 'cd_movies'
    data_config = config[data_setname]

    P = Pop(data_config)
    P.get_record_pop()
