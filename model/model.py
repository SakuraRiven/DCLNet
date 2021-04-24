import pdb
import cv2
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .resnet import resnet50
from .deform_resnet import resnet50
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class reduce_ch(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(reduce_ch, self).__init__()
        assert(len(in_chs) == len(out_chs) == 4)
        self.conv1 = nn.Conv2d(in_chs[0], out_chs[0], kernel_size=1)
        self.conv2 = nn.Conv2d(in_chs[1], out_chs[1], kernel_size=1)
        self.conv3 = nn.Conv2d(in_chs[2], out_chs[2], kernel_size=1)
        self.conv4 = nn.Conv2d(in_chs[3], out_chs[3], kernel_size=1)

    def forward(self, c2, c3, c4, c5):
        c5 = self.conv1(c5)
        c4 = self.conv2(c4)
        c3 = self.conv3(c3)
        c2 = self.conv4(c2)
        return c2, c3, c4, c5



class head(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(head, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1))

    def forward(self, x):
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, pretrain=False, backbone='50'):
        super(ResNetUNet, self).__init__()
        self.backbone = resnet50(pretrain) if backbone=='50' else resnet34(pretrain)
        in_chs = [2048, 1024, 512, 256] if backbone=='50' else [512, 256, 128, 64]
        out_chs = [i//2 for i in in_chs]
        self.redu_cls1 = reduce_ch(in_chs, out_chs)

        self.upconv1 = double_conv(out_chs[0] + out_chs[1], out_chs[1])
        self.upconv2 = double_conv(out_chs[1] + out_chs[2], out_chs[2])
        self.upconv3 = double_conv(out_chs[2] + out_chs[3], out_chs[3])

        self.head_cls = head(out_chs[-1], 1)
        self.head_rho = head(out_chs[-1], 4)
        self.head_theta = head(out_chs[-1], 4)


    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)

        c2, c3, c4, c5 = self.redu_cls1(c2, c3, c4, c5)
        y = F.interpolate(c5, size=c4.size()[2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, c4], dim=1)
        y1 = self.upconv1(y)

        y = F.interpolate(y1, size=c3.size()[2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, c3], dim=1)
        y2 = self.upconv2(y)

        y = F.interpolate(y2, size=c2.size()[2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, c2], dim=1)
        fuse = self.upconv3(y)

        cls = torch.sigmoid(self.head_cls(fuse))
        rho = F.relu(self.head_rho(fuse), inplace=True)
        theta = torch.sigmoid(self.head_theta(fuse)) * 2 * math.pi
        return cls, rho, theta


