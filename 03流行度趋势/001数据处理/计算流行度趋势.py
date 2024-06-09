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

class Pop:
    def __init__(self,data_config):
        self.config = data_config
        self.pop_time_gap = self.config['pop_time_gap']           #流行度间隔距离          movielens100 时间段为 1997.09-1998.04。
        self.train = []                     #uid  iid  rate  time  pop_value  is_pop
        self.test = []                      #uid  iid  rate  time  pop_value  is_pop

        self.item_num = 0
        self.user_num = 0
        self.min_time = 0
        self.max_time = 0
        self.get_data()


    def get_data(self):
        print('-----------加载数据---------')
        self.train = pd.read_csv(self.config['train_data'],names=['uid','iid','rate','time'],sep= self.config['split_gap'])
        self.train['pop_value'] = 1

        self.test = pd.read_csv(self.config['test_data'], names=['uid','iid','rate','time'],sep='\t')
        self.test['pop_value'] = 1
        self.min_time = self.train[['time']].min()['time']
        self.max_time = self.train[['time']].max()['time']
        self.user_num = int(  max(self.train[['uid']].max()['uid'],self.test[['uid']].max()['uid']))
        self.item_num = int( max(self.train[['iid']].max()['iid'],self.test[['iid']].max()['iid']))

    def get_record_pop(self):
        '''
        计算该项目时间点前后一段时间的流行度：该项目数/总项目数
        '''

        user_pop_tend = { i:[] for i in range(1,self.user_num+1)}

        for uid in range(1,self.user_num+1):
            start_time = self.min_time
            end_time = self.min_time + self.config['pop_time_gap']

            data =  self.train[self.train.uid == uid ]

            while(start_time < self.max_time):

                data1 = data[data['time'] >= start_time]

                data2 = data1[data1['time'] < end_time]

                if len(data2) == 0:
                    user_pop_tend[uid].append(0)

                else:
                    pop = data2[['rate']].mean()[0]
                    user_pop_tend[uid].append(pop)
                start_time += self.config['step_time']
                end_time += self.config['step_time']
        np.save(self.config['res_data']+'user_pop_tend_'+str(self.config['step_time'])+'.npy',user_pop_tend)


if __name__ == '__main__':
    config = {
        '100k': {
            'split_gap': ',',
            'step_time': 2 * 24 * 60 * 60,
            'pop_time_gap': 30 * 24 * 60 * 60,
            'train_data': '../../01数据集/100k/train_pop.csv',
            'test_data': '../../01数据集/100k/test.csv',
            'res_data': '../../01数据集/100k/',
        },
        'ml1m': {
            'split_gap': ',',
            'step_time': 17 * 24 * 60 * 60,
            'pop_time_gap': 30 * 24 * 60 * 60,
            'train_data': '../../01数据集/ml1m/train_pop.csv',
            'test_data': '../../01数据集/ml1m/test.csv',
            'res_data': '../../01数据集/ml1m/',
        },
        'cd_movies': {
            'step_time': 66 * 24 * 60 * 60,
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
