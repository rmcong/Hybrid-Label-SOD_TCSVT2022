import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F



class CB1(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(CB1, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        # x: [B, in_channels, H, W]
        y = self.conv(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = F.relu(y)
        return y

class CB3(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(CB3, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        # x: [B, in_channels, H, W]
        y = self.conv(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = F.relu(y)
        return y

class FC(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_relu=True):
        super(FC, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu 
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        # x: [B, in_channels]
        y = self.linear(x)
        if self.use_bn:
            y = self.bn(y)
        if self.use_relu:
            y = F.relu(y)
        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, sqz_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_1 = FC(in_channels, in_channels//sqz_ratio, True, True)
        self.fc_2 = FC(in_channels//sqz_ratio, in_channels, True, False) 
    def forward(self, ftr):
        # ftr: [B, C, H, W]
        avg_out = self.avg_pooling(ftr).squeeze(-1).squeeze(-1) # [B, C]
        max_out = self.max_pooling(ftr).squeeze(-1).squeeze(-1) # [B, C]
        avg_weights = self.fc_2(self.fc_1(avg_out)) # [B, C]
        max_weights = self.fc_2(self.fc_1(max_out)) # [B, C]
        weights = F.sigmoid(avg_weights + max_weights) # [B, C]
        return ftr * weights.unsqueeze(-1).unsqueeze(-1) + ftr
    
class GGD(nn.Module):
    def __init__(self, in_channels):
        super(GGD, self).__init__()
        self.channel_reduction = CB1(in_channels*2, in_channels, True, True)
        self.importance_estimator = nn.Sequential(ChannelAttention(in_channels),
                                    CB3(in_channels, in_channels//2, True, True),
                                    CB3(in_channels//2, in_channels//2, True, True),
                                    CB3(in_channels//2, in_channels, True, False), nn.Sigmoid())
    def forward(self, group_semantics, individual_features):
        # group_semantics & individual_features: [B, D=512, H=28, W=28]
        ftr_concat_reduc = self.channel_reduction(torch.cat((group_semantics, individual_features), dim=1)) # [B, D, H, W]
        P = self.importance_estimator(ftr_concat_reduc) # [B, D, H, W]
        co_saliency_features = group_semantics * P + individual_features * (1-P) # [B, D, H, W]
        return co_saliency_features
    
    

    