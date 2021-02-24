import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from .sparse_conv import SparseConv

class DilatedSparse3x3ConvBlock(nn.Module):
    """Layer to perform a 3x3 dilated convolution followed by BN and activation function
    """
    def __init__(self, in_channels, out_channels, activation_cls, dilation_rate):
        super(DilatedSparse3x3ConvBlock, self).__init__()

        self.sparse_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
        self.activation = activation_cls(inplace=True)

    def forward(self, input):
        x,m = self.sparse_conv(input)
        assert (m.size(1) == 1)
        out = self.activation(x)
        return out,m

class ParallelDilatedSparseConvolutions(nn.Module):

    def __init__(self, in_channels, out_channels, activation_cls, dilation_rates=[1,2,4,8], combination='sum'):
        super(ParallelDilatedSparseConvolutions, self).__init__()

        assert len(dilation_rates) > 0

        assert combination in ['sum', 'concat']

        bottleneck_channels = in_channels
        if combination == 'concat':
            assert in_channels % 4 == 0
            assert out_channels % len(dilation_rates) == 0
            bottleneck_channels = in_channels // 4
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
                f"dilated_pyramid_{dilation_rate}":  DilatedSparse3x3ConvBlock(bottleneck_channels, out_channels,
                                                                               activation_cls, dilation_rate)
            })

    def forward(self, input):

        x,m = input

        x = self.conv1(x)
        x = self.activation(x)

        pyramid_features = []
        for dilation_rate in self.dilation_rates:
            pyramid_feature, _ = self.dilated_pyramid[(f"dilated_pyramid_{dilation_rate}")]((x,m))
            pyramid_features.append(pyramid_feature)

        if self.combination == 'concat':
            out = torch.cat(pyramid_features, 1)
        else:
            out = torch.stack(pyramid_features)
            out = torch.sum(out, dim=0)

        return out, m