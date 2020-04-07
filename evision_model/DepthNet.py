# -*- coding: utf-8 -*-
"""
A network for predicting depth map.

based on "Depth from video in the wild", "SfmLearner-PyTorch" and "struct2depth"

code by jiafeng5513
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


class residual_block_A(nn.Module):
    def __init__(self, channel):
        super(residual_block_A, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Sequential(torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=2,
                                                   padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=2,
                                                   padding=1), nn.BatchNorm2d(channel))

    def forward(self, input_tensor):
        conv_out_1 = self.conv1(input_tensor)
        conv_out_2 = self.conv2(conv_out_1)
        pre_out = input_tensor + conv_out_2
        out = torch.nn.functional.relu(pre_out, inplace=True)
        return out


class residual_block_B(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(residual_block_B, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.stride = stride
        self.conv1 = nn.Sequential(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3,
                                                   stride=stride, padding=1), nn.BatchNorm2d(output_channel), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3,
                                                   stride=1, padding=1), nn.BatchNorm2d(output_channel))
        self.conv3 = nn.Sequential(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1,
                                                   stride=stride, padding=1))
        self.maxpool = nn.MaxPool2d((stride, stride), stride=(stride, stride))

    def forward(self, input_tensor):
        conv_out_1 = self.conv1(input_tensor)
        conv_out_2 = self.conv1(conv_out_1)
        if self.input_channel==self.output_channel:
            if self.stride == 1:
                shortcut = input_tensor
            else:
                shortcut =self.maxpool(input_tensor)
        else:
            shortcut = self.conv3(input_tensor)
        pre_out = shortcut + conv_out_2
        out = torch.nn.functional.relu(pre_out, inplace=True)
        return out


class encoder_module(nn.Module):
    def __init__(self):
        super(encoder_module, self).__init__()
        self.conv1 = nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                                    stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
        self.residual_block_A1 = residual_block_A(channel=64)
        self.residual_block_A2 = residual_block_A(channel=64)

        self.residual_block_first_B1 = residual_block_B(input_channel=64, output_channel=128, stride=2)
        self.residual_block_A3 = residual_block_A(channel=128)

        self.residual_block_first_B2 = residual_block_B(input_channel=128, output_channel=256, stride=2)
        self.residual_block_A4 = residual_block_A(channel=256)

        self.residual_block_first_B3 = residual_block_B(input_channel=256, output_channel=512, stride=2)
        self.residual_block_A5 = residual_block_A(channel=512)
        pass

    def forward(self, image):
        econv1 = self.conv1(image)
        maxpool_out = self.maxpool(econv1)
        residual_block_A1_out = self.residual_block_A1(maxpool_out)
        econv2 = self.residual_block_A2(residual_block_A1_out)

        residual_block_first_B1_out = self.residual_block_first_B1(econv2)
        econv3 = self.residual_block_A3(residual_block_first_B1_out)

        residual_block_first_B2_out = self.residual_block_first_B2(econv3)
        econv4 = self.residual_block_A4(residual_block_first_B2_out)

        residual_block_first_B3_out = self.residual_block_first_B3(econv4)
        econv5 = self.residual_block_A5(residual_block_first_B3_out)

        return econv5, econv4, econv3, econv2, econv1


class DepthNet(nn.Module):
    def __init__(self):
        self.encoder = encoder_module()
        self.uconv_5 = nn.Sequential(torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,stride=2), nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
        self.uconv_4 = nn.Sequential(torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,stride=2), nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
        self.uconv_3 = nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,stride=2), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
        self.uconv_2 = nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,stride=2), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
        self.uconv_1 = nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=2), nn.ReLU(inplace=True))
        self.conv_1 = nn.Sequential(torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1), nn.ReLU(inplace=True))
        self.conv_0 = nn.Sequential(torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1), nn.Softplus())

    def forward(self, image):
        econv5, econv4, econv3, econv2, econv1 = self.encoder(image)

        unconv5 = self.uconv_5(econv5)
        x5 = torch.cat([econv4, unconv5], dim=1)
        iconv5 = self.conv_5(x5)

        unconv4 = self.uconv_4(iconv5)
        x4 = torch.cat([econv3, unconv4], dim=1)
        iconv4 = self.conv_4(x4)

        unconv3 = self.uconv_3(iconv4)
        x3 = torch.cat([econv2, unconv3], dim=1)
        iconv3 = self.conv_3(x3)

        unconv2 = self.uconv_2(iconv3)
        x2 = torch.cat([econv1, unconv2], dim=1)
        iconv2 = self.conv_2(x2)

        unconv1 = self.uconv_1(iconv2)
        iconv1 = self.conv_1(unconv1)

        out = self.conv_0(iconv1)
        return out