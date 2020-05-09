# -*- coding: utf-8 -*-
"""
A network for predicting depth map.

based on "Depth from video in the wild", "SfmLearner-PyTorch" and "struct2depth"

U-Net liekd encoder-decoder architecture

code by jiafeng5513

NOTE:
    1. TensorFlow 的默认顺序是 [N H W C], PyTorch的默认顺序是 [N C H W]
    2. DepthNet输入一张张三通道图片，假设每张图片大小为[h,w],batch size =4，则输入张量为[4,3,h,w]
    3. 该文件的main函数仅供DepthNet进行shape检查
"""
import torch
import torch.nn as nn


class residual_block_A(nn.Module):
    def __init__(self, channel):
        super(residual_block_A, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Sequential(torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1,
                                                   padding=1), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1,
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
                                                   stride=stride))
        self.maxpool = nn.MaxPool2d((stride, stride), stride=(stride, stride))

    def forward(self, input_tensor):
        # input_tensor:[b,input_channel,h,w] 4,64,32,104
        conv_out_1 = self.conv1(input_tensor)   # [b,output_channel,h/2,w/2] 4,128,16,52
        conv_out_2 = self.conv2(conv_out_1)  # [b,output_channel,h/2,w/2] 4,128,16,52
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
                                    stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
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
        # image:[b,3,h,w]
        econv1 = self.conv1(image)  # [b,64,h/2,w/2]
        maxpool_out = self.maxpool(econv1)  # [b,64,h/4,w/4]
        residual_block_A1_out = self.residual_block_A1(maxpool_out)  # [b,64,h/4,w/4]
        econv2 = self.residual_block_A2(residual_block_A1_out)  # [b,64,h/4,w/4]

        residual_block_first_B1_out = self.residual_block_first_B1(econv2)  # [b,128,h/8,w/8]
        econv3 = self.residual_block_A3(residual_block_first_B1_out)  # [b,128,h/8,w/8]

        residual_block_first_B2_out = self.residual_block_first_B2(econv3)  # [b,256,h/16,w/16]
        econv4 = self.residual_block_A4(residual_block_first_B2_out)  # [b,256,h/16,w/16]

        residual_block_first_B3_out = self.residual_block_first_B3(econv4)  # [b,512,h/32,w/32]
        econv5 = self.residual_block_A5(residual_block_first_B3_out)  # [b,512,h/32,w/32]

        return econv5, econv4, econv3, econv2, econv1


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.encoder = encoder_module()
        self.uconv_5 = nn.Sequential(torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3,
                                                        stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3,
                                                    stride=1,padding=1), nn.ReLU(inplace=True))
        self.uconv_4 = nn.Sequential(torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3,
                                                        stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,
                                                    stride=1, padding=1), nn.ReLU(inplace=True))
        self.uconv_3 = nn.Sequential(torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3,
                                                        stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,
                                                    stride=1, padding=1), nn.ReLU(inplace=True))
        self.uconv_2 = nn.Sequential(torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3,
                                                        stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(torch.nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3,
                                                    stride=1, padding=1), nn.ReLU(inplace=True))
        self.uconv_1 = nn.Sequential(torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3,
                                                        stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True))
        self.conv_1 = nn.Sequential(torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                                                    stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv_0 = nn.Sequential(torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3,
                                                    stride=1, padding=1), nn.Softplus())

    def forward(self, image):
        # inage: [b,3,h,w] [4, 3, 128, 416]
        econv5, econv4, econv3, econv2, econv1 = self.encoder(image)
        # econv5:[b,512,h/32,w/32]
        # econv4:[b,256,h/16,w/16]  [4, 256,  8,  26]
        # econv3:[b,128,h/8,w/8]    [4, 128, 16,  52]
        # econv2:[b,64,h/4,w/4]     [4,  64, 32, 104]
        # econv1:[b,64,h/2,w/2]     [4,  64, 64, 208]
        unconv5 = self.uconv_5(econv5)              # [b,256,h/16,w/16]  [4, 256,  8,  26]
        x5 = torch.cat([econv4, unconv5], dim=1)    # [b,512,h/16,w/16]  [4, 512,  8,  26]
        iconv5 = self.conv_5(x5)                    # [b,256,h/16,w/16]  [4, 256,  8,  26]

        unconv4 = self.uconv_4(iconv5)              # [b,128,h/8,w/8]    [4, 128, 16,  52]
        x4 = torch.cat([econv3, unconv4], dim=1)    # [b,256,h/8,w/8]    [4, 256, 16,  52]
        iconv4 = self.conv_4(x4)                    # [b,128,h/8,w/8]    [4, 128, 16,  52]

        unconv3 = self.uconv_3(iconv4)              # [b,64,h/4,w/4]     [4,  64, 32, 104]
        x3 = torch.cat([econv2, unconv3], dim=1)    # [b,128,h/4,w/4]    [4,  128, 32, 104]
        iconv3 = self.conv_3(x3)                    # [b,64,h/4,w/4]     [4,  64, 32, 104]

        unconv2 = self.uconv_2(iconv3)              # [b,32,h/2,w/2]     [4,  32, 64, 208]
        x2 = torch.cat([econv1, unconv2], dim=1)    # [b,96,h/2,w/2]     [4,  96, 64, 208]
        iconv2 = self.conv_2(x2)                    # [b,32,h/2,w/2]     [4,  32, 64, 208]

        unconv1 = self.uconv_1(iconv2)              # [b,16,h,w]         [4,  16, 128, 416]
        iconv1 = self.conv_1(unconv1)               # [b,16,h,w]         [4,  16, 128, 416]

        out = self.conv_0(iconv1)                   # [b,1,h,w]          [4,  1, 128, 416]
        return out


if __name__ == '__main__':
    model = DepthNet()
    model = model.cuda()
    model.eval()

    image = torch.randn(4, 3, 128, 416)  # 输入尺寸 [N C H W]
    image = image.cuda()
    with torch.no_grad():
        depth_map = model(image)

    print(depth_map.shape)
