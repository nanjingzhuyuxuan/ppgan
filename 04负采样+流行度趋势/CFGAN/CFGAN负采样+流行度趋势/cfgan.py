# -*- coding: utf-8 -*-
"""
Author:
    Xuxin Zhang,xuxinz@qq.com
Reference: Chae D K , Kang J S , Kim S W , et al.
CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks[C]// the 27th ACM International Conference. ACM, 2018.

"""
import torch
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self, itemCount):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(itemCount, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        result = self.dis(data)
        return result


class generator(nn.Module):
    def __init__(self, itemCount,embedding):
        self.itemCount = itemCount
        self.embedding = embedding
        super(generator, self).__init__()
        self.gen1 = nn.Sequential(
            nn.Linear(self.itemCount, 1024),
            nn.ReLU(True)
        )
        self.gen2 = nn.Sequential(
            nn.Linear(self.embedding, 4),
            nn.ReLU(True)
        )
        self.gen3 = nn.Sequential(
            nn.Linear(1024+4, self.itemCount),
            nn.Tanh(),
        )
    def forward(self, noise,feature):
        result1 = self.gen1(noise)
        result_pop = self.gen2(feature)

        all_feature = torch.concat((result1,result_pop),1)
        result = self.gen3(all_feature)
        return result