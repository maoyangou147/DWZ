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
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=2, padding=1)  # 下采样到 (1, 128, 32, 32)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)          # 下采样到 (1, 256, 16, 16)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)          # 下采样到 (1, 512, 8, 8)

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        return [p1, p2, p3]