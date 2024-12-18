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
    def __init__(self, in_dims):
        super(FeatureFusionDecoder, self).__init__()

        c1_tgt_dims = 256
        c2_tgt_dims = 512
        c3_tgt_dims = 512

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dims, c1_tgt_dims, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(c1_tgt_dims),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dims, 384, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),# 1 384 32 32
            nn.Conv2d(384, c2_tgt_dims, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(c2_tgt_dims),
            nn.ReLU(),# 1 512 16 16
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c2_tgt_dims, c3_tgt_dims, 2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(c3_tgt_dims),
            nn.ReLU()# 1 512 8 8
        )


    def forward(self, x):
        # x_c3:1* x_c2:2* x_c1:4*
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)

        return [x1, x2, x3]