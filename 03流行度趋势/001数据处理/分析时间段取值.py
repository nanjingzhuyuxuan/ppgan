'''
计算用户的平均活跃时间间隔
'''

import pandas as pd
import json
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


class Pop:
    def __init__(self,data_config):
        self.config = data_config

        self.train = []                     #uid  iid  rate  time  pop_value  is_pop
        self.test = []                      #uid  iid  rate  time  pop_value  is_pop

        self.item_num = 0
        self.user_num = 0
        self.min_time = 0
        self.max_time = 0
        self.get_data()


    def get_data(self):
        print('-----------加载数据---------')
        self.train = pd.read_csv(self.config['train_data'],names=['uid','iid','rate','time'],sep=',')
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

        user_pop_time = { }
        user_pop_time_list =[]
        sum_last_time = 0
        uidlist =[]
        for uid in range(1,self.user_num+1):

            data =  self.train[self.train.uid == uid ]
            data[['time']] = data[['time']]/(24*60*60)
            data[['time']] =data[['time']].astype(int)

            num_action_day = len(data.groupby('time'))
            data_maxtime = data[['time']].max()
            data_mintime = data[['time']].min()
            diff_time = data_maxtime - data_mintime
            pop_last_time = diff_time / num_action_day
            user_pop_time.update({ uid :pop_last_time})
            sum_last_time+=pop_last_time
            uidlist.append(uid)
            user_pop_time_list.append(pop_last_time)
        print("平均活跃时间"+str(sum_last_time/self.user_num))


        plt.title('Average active time')
        plt.xlabel("User Id")
        plt.ylabel("Value (day)")
        plt.plot(uidlist, user_pop_time_list, marker='o')
        plt.savefig('precision_{0}.png'.format(data_setname))
        plt.show()

if __name__ == '__main__':
    config = {
        '100k': {

            'train_data': '../../01数据集/100k/train_pop.csv',
            'test_data': '../../01数据集/100k/test.csv',
        },
        'ml1m': {

            'train_data': '../../01数据集/ml1m/train_pop.csv',
            'test_data': '../../01数据集/ml1m/test.csv',
        },
        'cd_movies': {

            'train_data': '../../01数据集/cd_movies/train_pop.csv',
            'test_data': '../../01数据集/cd_movies/test.csv',
        },

    }
    data_setname = 'cd_movies'
    data_config = config[data_setname]
    P = Pop(data_config)
    P.get_record_pop()
#100k 24074.582689