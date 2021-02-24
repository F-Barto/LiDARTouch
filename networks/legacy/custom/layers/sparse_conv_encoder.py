from torch import nn

import numpy as np

from .sparse_conv import SparseConv
from .parallel_dilated_sparse_convolutions import ParallelDilatedSparseConvolutions

class SparseConv1x1(nn.Module):
    def __init__(self, in_channel, out_channel, activation_cls, bias=True):
        super(SparseConv1x1, self).__init__()
        self.sparse_conv = SparseConv(in_channel, out_channel, 1, padding=0, bias=bias)
        self.activation = activation_cls(inplace=True)

    def forward(self, input):
        x, m = self.sparse_conv(input)
        assert (m.size(1)==1)
        x = self.activation(x)
        return x, m

class SparseConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, activation_cls, stride=1, bias=True, mask_pool=False):
        super(SparseConv3x3, self).__init__()
        self.sparse_conv = SparseConv(in_channel, out_channel, 3,
                                      stride=stride, padding=1, bias=bias, mask_pool=mask_pool)
        self.activation = activation_cls(inplace=True)

    def forward(self, input):
        x, m = self.sparse_conv(input)
        assert (m.size(1)==1)
        x = self.activation(x)
        return x, m

class SparseConv7x7(nn.Module):
    def __init__(self, in_channel, out_channel, activation_cls, stride=1, bias=True, mask_pool=False):
        super(SparseConv7x7, self).__init__()
        self.sparse_conv = SparseConv(in_channel, out_channel, 7,
                                      stride=stride, padding=3, bias=bias, mask_pool=mask_pool)
        self.activation = activation_cls(inplace=True)

    def forward(self, input):
        x, m = self.sparse_conv(input)
        assert (m.size(1)==1)
        x = self.activation(x)
        return x, m

class SparseConvEncoder(nn.Module):
    def __init__(self, nb_blocks, activation, input_channels=1, dilation_rates=None, small=False, **kwargs):
        super(SparseConvEncoder, self).__init__()

        if dilation_rates is not None:
            assert len(dilation_rates) == len(nb_blocks)
            for list_dr in dilation_rates:
                assert isinstance(list_dr, list)
        else:
            dilation_rates = [None]*4

        self.input_channels = input_channels
        self.inplanes = 16

        self.activation = activation(inplace=True)

        self.first_conv = SparseConv7x7(self.input_channels, self.inplanes, activation, stride=2, mask_pool=True)

        self.small = small

        self.num_ch_enc = np.array([16, 16, 32, 64, 128])

        if self.small:
            self.num_ch_enc = np.array([16, 16, 32])

        ############### body ###############
        self.layer1 = self._make_layer(16, nb_blocks[0], activation, stride=2,
                                       dilation_rates=dilation_rates[0], **kwargs)
        self.layer2 = self._make_layer(32, nb_blocks[1], activation, stride=2,
                                       dilation_rates=dilation_rates[1], **kwargs)
        if not self.small:
            self.layer3 = self._make_layer(64, nb_blocks[2], activation, stride=2,
                                           dilation_rates=dilation_rates[2], **kwargs)
            self.layer4 = self._make_layer(128, nb_blocks[3], activation, stride=2,
                                           dilation_rates=dilation_rates[3], **kwargs)

    def _make_layer(self, planes, blocks, activation, stride=1, dilation_rates=None, **kwargs):

        layers = []

        if stride != 2:
            layers.append(SparseConv3x3(self.inplanes, planes, activation))
        else:
            layers.append(SparseConv3x3(self.inplanes, planes, activation, stride=stride, mask_pool=True))
        self.inplanes = planes

        if dilation_rates is not None:
            layers.append(ParallelDilatedSparseConvolutions(self.inplanes, planes, activation,
                                                            dilation_rates, **kwargs))

        for i in range(1, blocks):
                layers.append(SparseConv3x3(self.inplanes, planes, activation))

        return nn.Sequential(*layers)


    def forward(self, x):

        self.features = []

        m = (x > 0).detach().float()

        self.features.append(self.first_conv((x,m)))
        self.features.append(self.layer1(self.features[-1]))

        feature = self.layer2(self.features[-1])
        if not self.small:
            self.features.append(feature)
            self.features.append(self.layer3(self.features[-1]))
            self.features.append(self.layer4(self.features[-1]))
        else:
            self.features.append(feature)

        return self.features