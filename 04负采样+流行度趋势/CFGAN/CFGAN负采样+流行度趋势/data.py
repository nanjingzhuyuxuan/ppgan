# -*- coding: utf-8 -*-
"""
Author:
    Xuxin Zhang,xuxinz@qq.com
Reference: Chae D K , Kang J S , Kim S W , et al.
CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks[C]// the 27th ACM International Conference. ACM, 2018.

"""

from collections import defaultdict
import torch
import pandas as pd
import math,random

import matplotlib.pyplot as plt



def loadTrainingData(trainFile,splitMark):
    trainSet = defaultdict(dict)
    max_u_id = -1
    max_i_id = -1
    for line in open(trainFile):
        userId, itemId, rating ,_= line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        trainSet[userId].update({itemId:float(rating)})
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Training data loading done" )
    return trainSet,userCount,itemCount


def loadTestData(testFile,splitMark):
    testSet = defaultdict(dict)
    max_u_id = -1
    max_i_id = -1
    for line in open(testFile):
        userId, itemId, rating,_= line.strip().split(splitMark)
        userId = int(userId)
        itemId = int(itemId)
        testSet[userId].update({itemId:float(rating)})
        max_u_id = max(userId, max_u_id)
        max_i_id = max(itemId, max_i_id)
    userCount = max_u_id + 1
    itemCount = max_i_id + 1
    print("Test data loading done")
    return testSet,userCount,itemCount

import numpy as np
def to_Vectors(trainSet, userCount, itemCount, userList_test, mode,user_pop_tend):
    user_pop_tend_data = np.load(user_pop_tend, allow_pickle='TRUE').item()

    testMaskDict = defaultdict(lambda: [0] * itemCount)
    batchCount = userCount #改动  直接写成userCount
    if mode == "itemBased":#改动  itemCount userCount互换   batchCount是物品数
        userCount = itemCount
        itemCount = batchCount
        batchCount = userCount
    trainDict = defaultdict(lambda: [0] * itemCount)
    trainDict_pop = defaultdict(lambda: [0] * itemCount)
    for userId, i_list in trainSet.items():
        for iid,rating in i_list.items():
            testMaskDict[userId][iid] = -99999
            if mode == "userBased":
                trainDict[userId][iid] = 1.0
                trainDict_pop[userId][iid] = rating

            else:
                trainDict[iid][userId] = 1.0
                trainDict_pop[userId][iid] = rating
    # for userId, i_list in uninterest_mask.items():
    #     for iid in i_list:
    #         testMaskDict[userId][iid] = -99999

    testMaskVector = []
    for userId in userList_test:
        testMaskVector.append(testMaskDict[userId])
    return trainDict, torch.Tensor(testMaskVector), batchCount,trainDict_pop,user_pop_tend_data

def get_uninterest_item( trainSet,userCount,itemCount,path):
    uninterestdata  = np.load(path,allow_pickle=True).item()
    uninterest_mask = {}
    wake_uninterest_mask = {}

    for u in range(1,userCount):
        user_id_list = []  # 用户训练集项目，用于排除训练集id
        for item_i, pop_v in trainSet[u].items():
            user_id_list.append(item_i)
        all_item_index = random.sample(range(itemCount), itemCount)
        uninterest_item = uninterestdata[u]
        #排除不感兴趣的项目
        for j in uninterest_item:
            try:
                all_item_index.remove(j)
            except:continue
        #排除训练集项目
        for j in user_id_list:
            try:
                all_item_index.remove(j)
            except:
                continue
        #随机采样部分
        wake_uninterest_mask.update({u:all_item_index})
        uninterest_mask.update({u: uninterest_item})
    return uninterest_mask, wake_uninterest_mask


def item_pop(trainSet,userCount,itemCount):
    trainDict_pop = defaultdict(lambda: [0] * itemCount)
    for userId, i_list in trainSet.items():
        for iid,rating in i_list.items():
            trainDict_pop[userId][iid] = rating

    pop_list =[]
    for u,i_list in trainDict_pop.items():
        pop_list.append(i_list)
    pop_list_pd = pd.DataFrame(pop_list,columns=range(0,itemCount),index=range(1,userCount))

    item_pop_num_low = []   #低流行度
    item_pop_num_high = []  #高流行度
    #计算每个物品的流行度
    for i in range(0,itemCount):
        m = pop_list_pd[[i]].to_numpy()
        ma = m[m!=0]
        if len(ma) < 100:
            item_pop_num_low.append(i)
        if len(ma) > 300:
            item_pop_num_high.append(i)


    return item_pop_num_low,item_pop_num_high