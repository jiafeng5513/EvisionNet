# -*- coding: utf-8 -*-
"""
A network for predicting ego-motion, a 3D translation field and intrinsics.

based on "Depth from video in the wild" and "SfmLearner-PyTorch"

code by jiafeng5513

NOTE:
    1. TensorFlow 的默认顺序是 [N H W C], PyTorch的默认顺序是 [N C H W]
    2. MotionNet输入级联的两张三通道图片，假设每张图片大小为[h,w],batch size =4，
       则输入张量为[4,6,h,w],其中6=3*2
    3. 该文件的main函数仅供MotionNet进行shape检查
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


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
        # TODO:自适应卷积核策略
        # 当输入的w*h<3*3时，卷积核尺寸调整为1*1，否则为3*3
        self.conv1 = nn.Sequential(nn.Conv2d(motion_field_channel+layer_channel, max(4, layer_channel),
                                             stride=1, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(motion_field_channel+layer_channel, max(4, layer_channel),
                                             stride=1, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(max(4, layer_channel), max(4, layer_channel),
                                             stride=1, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(2*max(4, layer_channel), motion_field_channel,
                                             stride=1, kernel_size=1))

    def forward(self, motion_field, layer):
        # motion_field : [b,c1,h1,w1]
        # layer : [b,c2,h2,w2]
        N, C, H, W = layer.shape
        upsampled_motion_field = torch.nn.functional.interpolate(motion_field, size=[H, W])  # [b,c1,h2,w2]
        conv_input = torch.cat([upsampled_motion_field, layer], dim=1)  # [b,c1+c2,w2,h2]
        conv_out_1 = self.conv1(conv_input)  # [b,max(4,c2),h2,w2]
        conv_out_2 = self.conv2(conv_input)  # [b,max(4,c2),h2,w2]
        conv_out_3 = self.conv3(conv_out_2)  # [b,max(4,c2),h2,w2]
        conv_output = torch.cat([conv_out_1, conv_out_3], dim=1)  # [b,2*max(4,c2),h2,w2]
        conv_out_4 = self.conv4(conv_output)    # [b,c1,h2,w2]
        return upsampled_motion_field + conv_out_4  # [b,c1,h2,w2]


class IntrinsicsModule(nn.Module):
    def __init__(self):
        super(IntrinsicsModule,self).__init__()
        self.conv1 = nn.Conv2d(1024, 2, stride=1, kernel_size=1)
        self.conv2 = nn.Conv2d(1024, 2, stride=1, kernel_size=1)

    def forward(self, bottleneck, image_width, image_height):
        # bottleneck : [b, 1024, 1, 1]
        batch_size = bottleneck.shape[0]
        conv_out_1 = self.conv1(bottleneck)  # [b,2,1,1]
        conv_out_2 = self.conv2(bottleneck)  # [b,2,1,1]
        x1 = conv_out_1.squeeze()  # [b,2]
        x2 = conv_out_2.squeeze()  # [b,2]
        t = torch.Tensor([float(image_width), float(image_height)]).cuda()
        focal_lengths = x1*t  # [b,2]
        offsets = (x2+0.5)*t  # [b,2]
        foci = torch.diag_embed(focal_lengths)  # [b,2,2]
        offs = torch.unsqueeze(offsets, dim=-1)  # [b,2,1]
        pre_intrin = torch.cat([foci, offs], dim=2)  # [b,2,3]
        tail = torch.Tensor([0.0, 0.0, 1.0]).cuda()
        last_row = tail.repeat(batch_size, 1, 1)  # [b,1,3]
        intrinsic_mat = torch.cat([pre_intrin, last_row], dim=1)  # [b, 3, 3]
        return intrinsic_mat


class MotionNet(nn.Module):
    def __init__(self):
        super(MotionNet, self).__init__()
        # 7个3*3的2D卷积构成编码器
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        encode_channels = [6, 16, 32, 64, 128, 256, 512, 1024]
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[0], out_channels=encode_channels[1],
                                             kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[1], out_channels=encode_channels[2],
                                             kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[2], out_channels=encode_channels[3],
                                             kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[3], out_channels=encode_channels[4],
                                             kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[4], out_channels=encode_channels[5],
                                             kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[5], out_channels=encode_channels[6],
                                             kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=encode_channels[6], out_channels=encode_channels[7],
                                             kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        # 瓶颈衔接处
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=6, kernel_size=1, stride=1)  # TODO:in_channels is wrong!
        self.intrinsics_subnet = IntrinsicsModule()
        # 8个 refine构成解码器
        self.refine1 = RefineModule(motion_field_channel=3, layer_channel=encode_channels[7])  # translation, conv7
        self.refine2 = RefineModule(motion_field_channel=3, layer_channel=encode_channels[6])  # refine1_out, conv6
        self.refine3 = RefineModule(motion_field_channel=3, layer_channel=encode_channels[5])  # refine2_out, conv5
        self.refine4 = RefineModule(motion_field_channel=3, layer_channel=encode_channels[4])  # refine3_out, conv4
        self.refine5 = RefineModule(motion_field_channel=3, layer_channel=encode_channels[3])  # refine4_out, conv3
        self.refine6 = RefineModule(motion_field_channel=3, layer_channel=encode_channels[2])  # refine5_out, conv2
        self.refine7 = RefineModule(motion_field_channel=3, layer_channel=encode_channels[1])  # refine6_out, conv1
        self.refine8 = RefineModule(motion_field_channel=3, layer_channel=encode_channels[0])  # refine7_out, images

    def forward(self, images):
        # 编码器
        conv_out_1 = self.conv1(images)         # [b,16,h/2,w/2]
        conv_out_2 = self.conv2(conv_out_1)     # [b,32,h/4,w/4]
        conv_out_3 = self.conv3(conv_out_2)     # [b,64,h/8,w/8]
        conv_out_4 = self.conv4(conv_out_3)     # [b,128,h/16,w/16]
        conv_out_5 = self.conv5(conv_out_4)     # [b,256,h/32,w/32]
        conv_out_6 = self.conv6(conv_out_5)     # [b,512,h/64,w/64]
        conv_out_7 = self.conv7(conv_out_6)     # [b,1024,h/128,w/128]

        # 瓶颈
        bottleneck = torch.mean(conv_out_7, dim=[2, 3], keepdim=True)  # 在图片的两个坐标尺寸上求平均值 [b,1024,1,1]
        background_motion = self.conv8(bottleneck)  # [b,6,1,1]
        rotation = background_motion[:, :3, 0, 0]  # [B, 3]
        translation = background_motion[:, 3:, :, :]  # [B, 3, 1, 1],
        N, C, image_height, image_width = images.shape
        intrinsic_mat = self.intrinsics_subnet(bottleneck, image_height, image_width)   # [b,3,3]

        # 解码器
        refine_out_1 = self.refine1(translation, conv_out_7)  # [b,3,h/128,w/128]
        refine_out_2 = self.refine2(refine_out_1, conv_out_6)  # [b,3,h/64,w/64]
        refine_out_3 = self.refine3(refine_out_2, conv_out_5)  # [b,3,h/32,w/32]
        refine_out_4 = self.refine4(refine_out_3, conv_out_4)  # [b,3,h/16,w/16]
        refine_out_5 = self.refine5(refine_out_4, conv_out_3)  # [b,3,h/8,w/8]
        refine_out_6 = self.refine6(refine_out_5, conv_out_2)  # [b,3,h/4,w/4]
        refine_out_7 = self.refine7(refine_out_6, conv_out_1)  # [b,3,h/2,w/2]
        residual_translation = self.refine8(refine_out_7, images)  # [b,3,h,w]

        rot_scale = Variable(torch.Tensor([0.01]).cuda(), requires_grad=True)
        trans_scale = Variable(torch.Tensor([0.01]).cuda(), requires_grad=True)
        translation *= trans_scale
        residual_translation *= trans_scale
        rotation *= rot_scale
        # rotation : [b,3]
        # translation : [b,3,1,1]
        # residual_translation : [b,3,h,w]
        # intrinsic_mat : [b,3,3]
        return (rotation, translation, residual_translation, intrinsic_mat)


if __name__ == '__main__':

    model = MotionNet()
    model = model.cuda()
    model.eval()

    image = torch.randn(4, 6, 128, 416)  # 输入尺寸 [N C H W]
    image = image.cuda()
    with torch.no_grad():
        rotation, translation, residual_translation, intrinsic_mat = model(image)

    print(rotation.shape)
    print(translation.shape)
    print(residual_translation.shape)
    print(intrinsic_mat.shape)