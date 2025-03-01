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
'''

class FeatureAdapter(nn.Module):
    def __init__(self, c1,c2,c3,in_channels=256):
        super(FeatureAdapter, self).__init__()
        
        # 分支1：上采样到 (64, 128, 128)
        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, c1, kernel_size=4, stride=2, padding=1),  # 输出尺寸翻倍
            nn.ReLU(inplace=True)
        )
        
        # 分支2：保持分辨率 (128, 64, 64)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 分支3：下采样到 (256, 32, 32) 
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 输入 x 尺寸: (B, 256, 64, 64)
        p1 = self.branch1(x)  # (B, 64, 128, 128)
        p2 = self.branch2(x)  # (B, 128, 64, 64)
        p3 = self.branch3(x)  # (B, 256, 32, 32)
        return [p1, p2, p3]
