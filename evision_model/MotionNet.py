# -*- coding: utf-8 -*-
"""
A network for predicting ego-motion, a 3D translation field and intrinsics.

based on "Depth from video in the wild" and "SfmLearner-PyTorch"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
import torchvision.models
import collections
import math
# TensorFlow 的默认顺序是 [N H W C], PyTorch的默认顺序是 [N C H W]


class RefineModule(nn.Module):
    """
    motion_field ───────── [interpolate] ────────── upsampled_motion_field────────┬
    conv_input   ─┬─────────[conv1]────────┐                                      ├─── out
                  └───[conv2]────[conv3]───┴────conv_output─────[conv4]───────────┴
    定义参数:输入通道数和输出通道数
    运行参数:motion_field:前一级的解码器输入,layer:对位的编码器输入
    """
    def __init__(self, motion_field_channel, layer_channel):
        super(RefineModule, self).__init__()
        self.conv1 = nn.Conv2d(motion_field_channel, max(4, layer_channel), stride=1, kernel_size=3)
        self.conv2 = nn.Conv2d(motion_field_channel, max(4, layer_channel), stride=1, kernel_size=3)
        self.conv3 = nn.Conv2d(max(4, layer_channel), max(4, layer_channel), stride=1, kernel_size=3)
        self.conv4 = nn.Conv2d(max(4, layer_channel), motion_field_channel, stride=1, kernel_size=1)

    def forward(self, motion_field, layer):
        N, C, H, W = layer.shape
        upsampled_motion_field = torch.nn.functional.interpolate(motion_field, size=[H, W])
        conv_input = torch.cat([upsampled_motion_field, layer], dim=3)
        conv_out_1 = self.conv1(conv_input)
        conv_out_2 = self.conv2(conv_input)
        conv_out_3 = self.conv2(conv_out_2)
        conv_output = torch.cat([conv_out_1, conv_out_3], dim=-1)
        conv_out_4 = self.conv4(conv_output)
        return upsampled_motion_field + conv_out_4


class MotionNet(nn.Module):
    def __init__(self):
        super(MotionNet, self).__init__()
        # 7个3*3的2D卷积
        # TODO : weights_regularizer=layers.l2_regularizer(weight_reg)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        encode_channels = [3, 16, 32, 64, 128, 256, 512, 1024]
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[0], out_channels=encode_channels[1],
                                             kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[1], out_channels=encode_channels[2],
                                             kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[2], out_channels=encode_channels[3],
                                             kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[3], out_channels=encode_channels[4],
                                             kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[4], out_channels=encode_channels[5],
                                             kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[5], out_channels=encode_channels[6],
                                             kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[6], out_channels=encode_channels[7],
                                             kernel_size=3, stride=2), nn.ReLU(inplace=True))
        # 瓶颈衔接处
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=6, kernel_size=1, stride=1)  # TODO:in_channels is wrong!
        # 8个 refine
        refine_channels = []
        self.refine1 = RefineModule(motion_field_channel=6, layer_channel=2)  # translation, conv7
        self.refine2 = RefineModule(motion_field_channel=1, layer_channel=2)  # refine1_out, conv6
        self.refine3 = RefineModule(motion_field_channel=1, layer_channel=2)  # refine2_out, conv5
        self.refine4 = RefineModule(motion_field_channel=1, layer_channel=2)  # refine3_out, conv4
        self.refine5 = RefineModule(motion_field_channel=1, layer_channel=2)  # refine4_out, conv3
        self.refine6 = RefineModule(motion_field_channel=1, layer_channel=2)  # refine5_out, conv2
        self.refine7 = RefineModule(motion_field_channel=1, layer_channel=2)  # refine6_out, conv1
        self.refine8 = RefineModule(motion_field_channel=1, layer_channel=2)  # refine7_out, images

    def forward(self, images):
        conv_out_1 = self.conv1(images)
        conv_out_2 = self.conv2(conv_out_1)
        conv_out_3 = self.conv2(conv_out_2)
        conv_out_4 = self.conv2(conv_out_3)
        conv_out_5 = self.conv2(conv_out_4)
        conv_out_6 = self.conv2(conv_out_5)
        conv_out_7 = self.conv2(conv_out_6)

        # TODO:tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)
        bottleneck = torch.mean(conv_out_7, dim=[1, 2], keepdim=True)
        background_motion = self.conv8(bottleneck)
        rotation = background_motion[:, 0, 0, :3]
        translation = background_motion[:, :, :, 3:]



        pass
        # return out
