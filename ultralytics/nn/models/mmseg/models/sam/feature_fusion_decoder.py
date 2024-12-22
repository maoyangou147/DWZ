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

class FeatureFusionDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=[256, 512, 1024]):
        super(FeatureFusionDecoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        p3 = self.conv1(x)  # b*256*64*64 -> b*out_channels[0]*64*64
        p4 = self.conv2(p3)  # b*out_channels[0]*64*64 -> b*out_channels[1]*32*32
        p5 = self.conv3(p4)  # b*out_channels[1]*32*32 -> b*out_channels[2]*16*16
        return [p3, p4, p5]
