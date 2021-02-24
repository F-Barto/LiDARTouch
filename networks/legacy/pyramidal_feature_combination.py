"""

Inspired by ASPP, PSPNet and Inception modules

Pyramidal Feature Combination
"""

import torch
from torch import nn
from collections import OrderedDict


class Dilated3x3ConvBlock(nn.Module):
    """Layer to perform a 3x3 dilated convolution followed by BN and activation function
    """
    def __init__(self, in_channels, out_channels, activation_cls, dilation_rate):
        super(Dilated3x3ConvBlock, self).__init__()


        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation_cls(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out

class PyramidalFeatureCombination(nn.Module):

    def __init__(self, in_channels, out_channels, activation_cls, dilation_rates=[1,2,8,16], combination='sum'):
        super(PyramidalFeatureCombination, self).__init__()

        assert len(dilation_rates) > 0

        assert combination in ['sum', 'concat']

        bottleneck_channels = in_channels
        if combination == 'concat':
            assert in_channels % len(dilation_rates) == 0
            assert out_channels % len(dilation_rates) == 0
            bottleneck_channels = in_channels // len(dilation_rates)
            out_channels = out_channels // len(dilation_rates)

        self.combination = combination
        self.dilation_rates = dilation_rates

        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = norm_layer(bottleneck_channels)
        self.activation = activation_cls(inplace=True)

        self.dilated_pyramid = nn.ModuleDict()
        for dilation_rate in self.dilation_rates:
            self.dilated_pyramid.update({
                f"dilated_pyramid_{dilation_rate}":  Dilated3x3ConvBlock(bottleneck_channels, out_channels,
                                                                         activation_cls, dilation_rate)
            })

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        pyramid_features = []
        for dilation_rate in self.dilation_rates:
            pyramid_feature = self.dilated_pyramid[(f"dilated_pyramid_{dilation_rate}")](out)
            pyramid_features.append(pyramid_feature)

        if self.combination == 'concat':
            out = torch.cat(pyramid_features, 1)
        else:
            out = torch.stack(pyramid_features)
            out = torch.sum(out, dim=0)

        return out