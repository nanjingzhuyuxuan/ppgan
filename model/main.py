# -*- coding: utf-8 -*-
"""
当前版本：22-12-16
基于简单强弱负采样策略，若该用户喜欢流行度高的项目，则其强负面样本为流行度低的物品；若该用户喜欢流行度低的项目，则强负面样本为流行度高的样本
实验结果记录：
100k   max_m = 0.3  min_m = 0.15
epoch:333,precision:0.3985169491525395,precision:0.3354872881355932,ndcg5:0.4217615967765171,ndcg10:0.369870742916149,precisions_l_5:0.019491525423728798,precisions_l_10:0.026800847457627158

一层隐含层
epoch:163,precision:0.40932203389830174,precision:0.3373940677966104,ndcg5:0.43424064304894855,ndcg10:0.37555642843899273,precisions_l_5:0.013771186440677964,precisions_l_10:0.018961864406779657

max 0.3  min 0.2  0.1 0.7\
epoch:154,precision:0.39915254237287867,precision:0.3343220338983051,ndcg5:0.41786133446075924,ndcg10:0.36658904229160066,precisions_l_5:0.016737288135593206,precisions_l_10:0.02108050847457627

"""

import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import data
import cfgan

import warnings

warnings.filterwarnings("ignore")


def select_negative_items(realData, num_pm, uninterest_list,wake_uninterest_list):  # num_pm 50  num_zr50
    '''
    realData : n-dimensional indicator vector specifying whether u has purchased each item i
    num_pm : num of negative items (partial-masking) sampled on the t-th iteration                 在第t次迭代中采样的负项数（部分掩蔽）
    num_zr : num of negative items (zeroreconstruction regularization) sampled on the t-th iteration           在第t次迭代中采样的负项数（零，重建正则化）
    '''
    data = np.array(realData.cpu())
    n_items_pm = np.zeros_like(data)  # 构造一个像data一样形状的0矩阵          (32, 16028)

    for i in range(data.shape[0]):  # 32*1683
        uninterest_items = uninterest_list[i]
        wake_uninterest_items = wake_uninterest_list[i]
        # all_items = wake_uninterest_items + uninterest_items

        random.shuffle(wake_uninterest_items)
        if len(wake_uninterest_items)<num_pm:
            random_items_index_list = wake_uninterest_items
        else:
            random_items_index_list = random.sample(wake_uninterest_items, num_pm)
        negative_items_index_list = random_items_index_list + uninterest_items

        n_item_index_pm = negative_items_index_list  # 对项目顺序去前num_pr个
        n_items_pm[i][n_item_index_pm] = 1.0  # (512, 16028)

    return n_items_pm


import math

def computeTopN(groundTruth, result, topN, item_pop_num,item_pop_num_high):
    # 准确率
    result = result.tolist()
    for i in range(len(result)):
        result[i] = (result[i], i)
    result.sort(key=lambda x: x[0], reverse=True)

    hit = 0
    hit_low = 0
    hit_high = 0
    for i in range(topN):
        if (result[i][1] in groundTruth):
            if result[i][1] in item_pop_num:
                hit_low += 1
            if result[i][1] in item_pop_num_high:
                hit_high+=1
            hit = hit + 1

    p = hit / topN

    # NDCG
    DCG = 0
    IDCG = 0
    # 1 = related, 0 = unrelated
    for i in range(topN):
        if (result[i][1] in groundTruth):
            DCG += 1.0 / math.log(i + 2)

    for i in range(topN):
        IDCG += 1.0 / math.log(i + 2)
    ndcg = DCG / IDCG
    return p, ndcg, hit_low / topN,hit_high/topN

def computeTopN_longtail(groundTruth, result, topN, item_pop_num):
    # 准确率
    result = result.tolist()
    for i in range(len(result)):
        result[i] = (result[i], i)
    result.sort(key=lambda x: x[0], reverse=True)
    #选择top长尾
    hit = 0

    top_longtail = []
    for i in range(len(result)):
        if len(top_longtail) > topN:
            break
        elif result[i][1] in item_pop_num:
            top_longtail.append( result[i][1])
        else:
            continue

    for i in range(topN):
        if (top_longtail[i] in groundTruth):
            hit = hit + 1

    p = hit / topN

    # NDCG
    DCG = 0
    IDCG = 0
    # 1 = related, 0 = unrelated
    for i in range(topN):
        if (top_longtail[i] in groundTruth):
            DCG += 1.0 / math.log(i + 2)

    for i in range(topN):
        IDCG += 1.0 / math.log(i + 2)
    ndcg = DCG / IDCG
    return p, ndcg




def main(userCount, itemCount, trainSet, testSet, trainVector, trainMaskVector, epochCount, pro_PM, alpha,
          item_pop_num, uninterest_mask,wake_uninterest_mask,item_pop_num_high,user_pop_tend_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_precision = np.zeros((1, 2))
    embedding = len(user_pop_tend_data[1])
    # Build the generator and discriminator
    G = cfgan.generator(itemCount,embedding).cuda().to(device)
    D = cfgan.discriminator(itemCount).cuda().to(device)
    regularization = nn.MSELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    G_step = 5
    D_step = 2
    batchSize_G = 32
    batchSize_D = 32

    for epoch in range(epochCount):

        # ---------------------
        #  Train Generator
        # ---------------------

        for step in range(G_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_G - 1)
            uidList = list(trainSet)[leftIndex:leftIndex + batchSize_G]  # 32, 1683
            popData = [user_pop_tend_data[i] for i in uidList]

            realData = [trainVector[i] for i in uidList]
            uninterest_list = [uninterest_mask[i] for i in uidList]
            wake_uninterest_list = [wake_uninterest_mask[i] for i in uidList]

            popData = Variable(torch.tensor(popData,dtype=torch.float32))
            realData = Variable(torch.tensor(realData))
            eu = realData.cuda().to(device)  # 是一个m维指示向量，表示用户是否对项目进行了评分

            # Select a random batch of negative items for every user
            n_items_pm = select_negative_items(realData, pro_PM, uninterest_list,wake_uninterest_list)
            ku_zp = Variable(torch.tensor(n_items_pm )).cuda().to(device)

            realData_zp = Variable(torch.ones_like(realData)).cuda().to(device) * eu + Variable(torch.zeros_like(realData)).cuda().to(device) * ku_zp
            realData = realData.cuda().to(device)
            popData = popData.cuda().to(device)
            # Generate a batch of new purchased vector
            fakeData = G(realData,popData).cuda().to(device)
            fakeData_ZP = fakeData * (eu + ku_zp)
            fakeData_result = D(fakeData_ZP)

            # Train the discriminator
            g_loss = np.mean(np.log(1. - fakeData_result.detach().cpu().numpy() + 10e-5)) + alpha * regularization(
                fakeData_ZP, realData_zp)
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for step in range(D_step):
            # Select a random batch of purchased vector
            leftIndex = random.randint(1, userCount - batchSize_D - 1)
            uidList = list(trainSet)[leftIndex:leftIndex + batchSize_G]  # 32, 1683
            realData = [trainVector[i] for i in uidList]
            popData = [user_pop_tend_data[i] for i in uidList]

            uninterest_list = [uninterest_mask[i] for i in uidList]
            wake_uninterest_list = [wake_uninterest_mask[i] for i in uidList]
            popData = Variable(torch.tensor(popData,dtype=torch.float32))
            realData = Variable(torch.tensor(realData))
            eu = realData.cuda().to(device)
            popData = popData.cuda().to(device)
            # Select a random batch of negative items for every user
            n_items_pm = select_negative_items(realData, pro_PM, uninterest_list,wake_uninterest_list)
            ku = Variable(torch.tensor(n_items_pm)).cuda().to(device)
            realData = realData.cuda().to(device)
            # Generate a batch of new purchased vector
            fakeData = G(realData,popData).cuda().to(device)
            fakeData_ZP = fakeData * (eu + ku)

            # Train the discriminator
            fakeData_result = D(fakeData_ZP)
            realData_result = D(realData)
            d_loss = -np.mean(np.log(realData_result.detach().cpu().numpy() + 10e-5) +
                              np.log(1. - fakeData_result.detach().cpu().numpy() + 10e-5)) + 0 * regularization(
                fakeData_ZP, realData_zp)
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

        if (epoch % 1 == 0):
            n_user = len(testSet)
            precisions_5 = 0
            precisions_10 = 0
            ndcg_5 = 0
            ndcg_10 = 0
            index = 0
            precisions_l_5 = 0
            precisions_l_10 = 0
            precisions_h_5 = 0
            precisions_h_10 = 0
            #三类人群
            like_pop_p_5 = [0,0,0]  # like_pop_low_p_5  like_pop_high_p_5  like_pop_normal_p_5
            like_pop_p_10 = [0,0,0]  # like_pop_low_p_5  like_pop_high_p_5  like_pop_normal_p_5
            like_pop_ndcg_5 = [0,0,0]  # like_pop_low_p_5  like_pop_high_p_5  like_pop_normal_p_5
            like_pop_ndcg_10 = [0,0,0]  # like_pop_low_p_5  like_pop_high_p_5  like_pop_normal_p_5

            # 长尾
            precisions_longtail_5, ndcg_longtail_5 = 0,0
            precisions_longtail_10, ndcg_longtail_10 = 0,0

            for testUser in testSet.keys():
                realData = Variable(torch.tensor(trainVector[testUser], dtype=torch.float32))
                data = Variable(copy.deepcopy(realData)).cuda().to(device)
                #  Exclude the purchased vector that have occurred in the training set
                popData = Variable(torch.tensor(user_pop_tend_data[testUser], dtype=torch.float32))
                popData = Variable(copy.deepcopy(popData)).cuda().to(device)
                result = G(data.reshape(1, itemCount),popData.reshape(1, embedding)) + Variable(copy.deepcopy(trainMaskVector[index])).cuda().to(
                    device)
                result = result.reshape(itemCount)

                precision, ndcg, precisions_l,precisions_h = computeTopN(testSet[testUser], result, 5, item_pop_num,item_pop_num_high)

                # precisions_longtail,ndcg_longtail = computeTopN_longtail(testSet[testUser], result, 5, item_pop_num)
                precisions_5 += precision
                precisions_l_5 += precisions_l
                precisions_h_5 += precisions_h
                ndcg_5 += ndcg
                # precisions_longtail_5 += precisions_longtail
                # ndcg_longtail_5 += ndcg_longtail


                # precisions_5 += precisions_longtail
                # ndcg_5 += ndcg_longtail
                #
                # if testUser in like_pop_low:
                #     like_pop_p_5[0] += precisions_longtail
                #     like_pop_ndcg_5[0] += ndcg_longtail
                # elif testUser in like_pop_high:
                #     like_pop_p_5[1] += precisions_longtail
                #     like_pop_ndcg_5[1] += ndcg_longtail
                # else:
                #     like_pop_p_5[2] += precisions_longtail
                #     like_pop_ndcg_5[2] += ndcg_longtail
                # TOP10
                precision, ndcg, precisions_l ,precisions_h= computeTopN(testSet[testUser], result, 10, item_pop_num,item_pop_num_high)
                # precisions_longtail,ndcg_longtail = computeTopN_longtail(testSet[testUser], result, 10, item_pop_num)
                precisions_10 += precision
                ndcg_10 += ndcg
                precisions_l_10 += precisions_l
                precisions_h_10 +=precisions_h
                # precisions_longtail_10 += precisions_longtail
                # ndcg_longtail_10 += ndcg_longtail

                # precisions_10 += precisions_longtail
                # ndcg_10 += ndcg_longtail

                #
                # if testUser in like_pop_low:
                #     like_pop_p_10[0] += precisions_longtail
                #     like_pop_ndcg_10[0] += ndcg_longtail
                # elif testUser in like_pop_high:
                #     like_pop_p_10[1] += precisions_longtail
                #     like_pop_ndcg_10[1] += ndcg_longtail
                # else:
                #     like_pop_p_10[2] += precisions_longtail
                #     like_pop_ndcg_10[2] += ndcg_longtail
                index += 1

            precisions_5 = precisions_5 / n_user
            precisions_10 = precisions_10 / n_user
            precisions_l_10 = precisions_l_10 / n_user
            precisions_l_5 = precisions_l_5 / n_user
            ndcg_5 = ndcg_5 / n_user
            ndcg_10 = ndcg_10 / n_user
            precisions_h_5 = precisions_h_5 /n_user
            precisions_h_10 = precisions_h_10 / n_user
            #长尾
            # precisions_longtail_5 = precisions_longtail_5/ n_user
            # ndcg_longtail_5 = ndcg_longtail_5/ n_user
            # precisions_longtail_10 = precisions_longtail_10/ n_user
            # ndcg_longtail_10 = ndcg_longtail_10/ n_user

            # like_pop_p_5[0],like_pop_p_5[1],like_pop_p_5[2] = like_pop_p_5[0]/len(like_pop_low),like_pop_p_5[1]/len(like_pop_high),like_pop_p_5[2]/(n_user-len(like_pop_low)-len(like_pop_high))
            # like_pop_ndcg_5[0],like_pop_ndcg_5[1],like_pop_ndcg_5[2] = like_pop_ndcg_5[0]/len(like_pop_low),like_pop_ndcg_5[1]/len(like_pop_high),like_pop_ndcg_5[2]/(n_user-len(like_pop_low)-len(like_pop_high))
            # like_pop_p_10[0],like_pop_p_10[1],like_pop_p_10[2] = like_pop_p_10[0]/len(like_pop_low),like_pop_p_10[1]/len(like_pop_high),like_pop_p_10[2]/(n_user-len(like_pop_low)-len(like_pop_high))
            # like_pop_ndcg_10[0],like_pop_ndcg_10[1],like_pop_ndcg_10[2] = like_pop_ndcg_10[0]/len(like_pop_low),like_pop_ndcg_10[1]/len(like_pop_high),like_pop_ndcg_10[2]/(n_user-len(like_pop_low)-len(like_pop_high))


            # print('epoch:{},precision:{},precision_10:{},ndcg5:{},ndcg10:{},precisions_l_5:{},precisions_l_10:{},precisions_h_5:{},precisions_h_10:{},'
            #       'like_pop_low_p_5:{},  like_pop_high_p_5:{},like_pop_normal_p_5:{},'
            #       'like_pop_low_p_10:{},  like_pop_high_p_10:{},like_pop_normal_p_10:{},'
            #       'like_pop_ndcg_5:{},like_pop_ndcg_5:{},like_pop_ndcg_5:{},like_pop_ndcg_10:{}, like_pop_ndcg_10:{}, like_pop_ndcg_10:{},'.format(
            #     epoch, precisions_5, precisions_10, ndcg_5, ndcg_10, precisions_l_5, precisions_l_10,precisions_h_5,precisions_h_10,
            #
            #     like_pop_p_5[0], like_pop_p_5[1], like_pop_p_5[2],like_pop_p_10[0],like_pop_p_10[1],like_pop_p_10[2],
            #
            #     like_pop_ndcg_5[0],like_pop_ndcg_5[1],like_pop_ndcg_5[2],like_pop_ndcg_10[0], like_pop_ndcg_10[1], like_pop_ndcg_10[2],
            # ))
            print('epoch:{},precision:{},precision_10:{},ndcg5:{},ndcg10:{},precisions_l_5:{},precisions_l_10:{},precisions_h_5:{},precisions_h_10:{}'.format(
                epoch, precisions_5, precisions_10, ndcg_5, ndcg_10, precisions_l_5, precisions_l_10,precisions_h_5,precisions_h_10,
            ))




    return result_precision


if __name__ == '__main__':
    data_name = '100k'
    config = {
        '100k': {
            'train': '../../../01数据集/100k/weight_train_pop.csv',
            'test': '../../../01数据集/100k/test.csv',
            'uninterest': '../../../01数据集/100k/user_uninterest_item_weight_5.npy',
            'pop_tend': '../../../01数据集/100k/user_pop_tend_691200.npy',
        },
        'ml1m': {
            'train': '../../../01数据集/ml1m/weight_train_pop.csv',
            'test': '../../../01数据集/ml1m/test.csv',





        },
        'cd_movies': {
            'train': '../../../01数据集/cd_movies/all_train_pop.csv',
            'test': '../../../01数据集/cd_movies/test.csv',
            'uninterest': '../../../01数据集/cd_movies/user_uninterest_item_weight_7.npy',
            'pop_tend': '../../../01数据集/100k/m_user_pop_tend_604800.npy',
        },

    }

    data_config = config[data_name]
    topN = 5
    epochs = 1000
    pro_ZR = 100
    pro_PM = 100
    alpha = 0.1

    trainSet, train_use, train_item = data.loadTrainingData(data_config['train'], ",")
    testSet, test_use, test_item = data.loadTestData(data_config['test'], "\t")
    userCount = max(train_use, test_use)  # 7854
    itemCount = max(train_item, test_item)  # 16185

    item_pop_num_low,item_pop_num_high = data.item_pop(trainSet, userCount, itemCount)
    uninterest_mask ,wake_uninterest_mask = data.get_uninterest_item( trainSet,userCount,itemCount,data_config['uninterest'])
    userList_test = list(testSet.keys())
    trainVector, trainMaskVector, batchCount, trainDict_pop,user_pop_tend_data = data.to_Vectors(trainSet, userCount, itemCount,
                                                                                 userList_test, "userBased",
                                                                                 data_config['pop_tend'])

    result_precision = main(userCount, itemCount, trainSet, testSet, \
                            trainVector, trainMaskVector, epochs, pro_PM, alpha,
                            item_pop_num_low, uninterest_mask, wake_uninterest_mask,item_pop_num_high,user_pop_tend_data)
    result_precision = result_precision[1:, ]



