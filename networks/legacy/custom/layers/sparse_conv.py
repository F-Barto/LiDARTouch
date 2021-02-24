# Adapted from https://github.com/yxgeee/DepthComplete/blob/release/models/SparseConvNet.py


import torch
from torch import nn
from torch.nn import functional as F

class SparseConv(nn.Module):
    # Convolution layer for sparse data
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 mask_pool=False):
        super(SparseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.if_bias = bias
        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)

        self.mask_pool = mask_pool
        if self.mask_pool:
            self.pool = SparseMaxPool(kernel_size, stride=stride, padding=padding, dilation=dilation)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x, m = input
        mc = m.expand_as(x)
        x = x * mc
        x = self.conv(x)

        ones = torch.ones_like(self.conv.weight)
        mc = F.conv2d(mc, ones, bias=None, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        mc = torch.clamp(mc, min=1e-5)
        mc = 1. / mc
        x = x * mc
        if self.if_bias:
            x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)

        if self.mask_pool:
            m = self.pool(m)

        return x, m

class SparseMaxPool(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(SparseMaxPool, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.pool.require_grad = False

    def forward(self, m):
        m = self.pool(m)
        return m