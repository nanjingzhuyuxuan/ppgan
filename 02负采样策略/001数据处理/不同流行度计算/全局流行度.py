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
config = {
    '100k':{
        'pop_time_gap':30*24*60*60 ,
        'train_data':'../../../01数据集/100k/train.csv',
        'test_data':'../../../01数据集/100k/test.csv',
        'train_pop_txt':'../../../01数据集/100k/all_train_pop.csv',

    },
    'ml1m': {
        'pop_time_gap': 30 * 24 * 60 * 60,
        'train_data': '../../../01数据集/ml1m/train.csv',
        'test_data': '../../../01数据集/ml1m/test.csv',
        'train_pop_txt': '../../../01数据集/ml1m/all_train_pop.csv',

    },
    'cd_movies': {
        'pop_time_gap': 180 * 24 * 60 * 60,
        'train_data': '../../../01数据集/cd_movies/train.csv',
        'test_data': '../../../01数据集/cd_movies/test.csv',
        'train_pop_txt': '../../../01数据集/cd_movies/all_train_pop.csv',

    },
    'goodreads': {
        'pop_time_gap': 180 * 24 * 60 * 60,
        'train_data': '../../../01数据集/goodreads/train.csv',
        'train_pop_txt': '../../../01数据集/goodreads/all_train_pop.csv',

    }
}
class Pop:
    def __init__(self,data_config):
        self.config = data_config
        self.pop_time_gap = self.config['pop_time_gap']           #流行度间隔距离          movielens100 时间段为 1997.09-1998.04。
        self.train = []                     #uid  iid  rate  time  pop_value  is_pop
        self.test = []                      #uid  iid  rate  time  pop_value  is_pop
        self.ispop = []                     #每条记录  是否是流行度项目  1/0
        self.item_pop_time = {}             #项目流行度时间、均值、标准差
        self.item_num = 0
        self.user_num = 0
        self.get_data()


    def get_data(self):
        print('-----------加载数据---------')
        self.train = pd.read_csv(self.config['train_data'],names=['uid','iid','time'],sep=',')
        self.train['pop_value'] = 1

        # self.test = pd.read_csv(self.config['test_data'], names=['uid','iid','rate','time'],sep='\t')
        # self.test['pop_value'] = 1

        # self.user_num = int(  max(self.train[['uid']].max()['uid'],self.test[['uid']].max()['uid']))
        # self.item_num = int( max(self.train[['iid']].max()['iid'],self.test[['iid']].max()['iid']))
        # print(self.user_num,self.item_num)
        # self.item_pop_time = {i:{} for i in range(1,self.item_num+1)}

    def get_record_pop(self):
        '''

        '''

        for index,item in self.train.iterrows():

            iid = item['iid']
            #找到+-pop_time_gap  的数据集
            data = self.train[self.train.iid <= iid ]
            len_data = len(data)
            pop = len(data[data['iid']==iid])/len_data
            self.train.loc[index,'pop_value'] = round(pop,5)

        self.train = self.train[['uid','iid','pop_value','time']]
        self.train.to_csv(self.config['train_pop_txt'],index=False)

        plot(self.train)
import matplotlib.pyplot as plt
def plot(data):
    x = data['pop_value'].tolist()

    plt.rcParams['axes.unicode_minus']=False#显示负号\n",
    plt.figure(figsize=(6,4))## 设置画布\n",
    plt.hist(x,bins=100)

    plt.title('100k-all distribution')
    plt.xlabel('rate')
    plt.ylabel('Probability')

    plt.show()

if __name__ == '__main__':
    data_setname = 'goodreads'
    data_config = config[data_setname]
    P = Pop(data_config)
    P.get_record_pop()
