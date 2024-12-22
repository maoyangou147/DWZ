import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

'''
yolov8l
1 256 32 32 
1 512 16 16
1 512 8 8
p:1 32 64 64 
mc: 1 32 1344

sam
c1 1 128 256 256  -->1 512 16 16
c2 1 128 128 128  -->1 512 8 8
c3 1 256 64 64    -->1 256 32 32 
'''

class FeatureAdapter(nn.Module):
    def __init__(self, in_channels=256):
        super(FeatureAdapter, self).__init__()
        
        # 定义卷积层以调整通道数和分辨率
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1)  # 下采样到 (1, 128, 32, 32)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)          # 下采样到 (1, 256, 16, 16)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)          # 下采样到 (1, 512, 8, 8)

        # 可选：添加激活函数（如 ReLU）和归一化层（如 BatchNorm2d）
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)

    def forward(self, x):
        # 第一步：从 (1, 256, 64, 64) 到 (1, 128, 32, 32)
        p1 = self.conv1(x)
        p1 = self.bn1(p1)
        p1 = self.relu(p1)

        # 第二步：从 (1, 128, 32, 32) 到 (1, 256, 16, 16)
        p2 = self.conv2(p1)
        p2 = self.bn2(p2)
        p2 = self.relu(p2)

        # 第三步：从 (1, 256, 16, 16) 到 (1, 512, 8, 8)
        p3 = self.conv3(p2)
        p3 = self.bn3(p3)
        p3 = self.relu(p3)

        return [p1, p2, p3]
