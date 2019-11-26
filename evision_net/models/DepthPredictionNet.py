# -*- coding: utf-8 -*-
"""
Depth prediction net

based on "Depth from video in the wild"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import torchvision.models
import collections
import math


class DepthPredictionNet(nn.Module):
    def __init__(self, normalizer_fn=None):
        super(DepthPredictionNet, self).__init__()

        # 定义encode_resnet

        self.global_pooling = nn.AvgPool2d([8,4], stride=[8,4], padding=(4, 2))   # [N,2048,3,14]
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048 * 3 * 14, 512)  # 3 * 14
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积
        # self.upsample = nn.UpsamplingBilinear2d(size=(16, 52))
        #self.upsample = nn.functional.interpolate(size=(16, 52), mode='bilinear', align_corners=True)
        weights_init(self.modules(), 'xavier')

    def forward(self, x):
        # print('x size:', x.size())
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * 3 * 14)  # 3 * 14
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        out = nn.functional.interpolate(input=x5, size=(16, 52), mode='bilinear', align_corners=True)  # UpsamplingBilinear2d
        return out


