# code from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/resnet/depth_decoder.py

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from networks.legacy.monodepth2.layers.common import ConvBlock, Conv3x3
from .packing import UnpackLayerConv3d



class DepthPackDecoder(nn.Module):
    def __init__(self, num_ch_enc, activation, scales=range(3), num_output_channels=1,
                 concat_skips=True, uncertainty=False, refine_pred=False):
        super(DepthPackDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.concat_skips = concat_skips
        self.scales = scales
        self.uncertainty = uncertainty
        self.refine_pred = refine_pred

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32,64,128])

        num_3d_feat = 4

        # decoder
        self.convs = OrderedDict()

        self.convs["first_upconv"] = ConvBlock(self.num_ch_enc[-1], self.num_ch_dec[-1], activation)
        self.convs["first_unpack"] = UnpackLayerConv3d(self.num_ch_dec[-1], self.num_ch_dec[-2], 3, d=num_3d_feat)

        for i in range(1, -1, -1):  # [1, 0]
            # upconv pre upsampling
            num_ch_in = self.num_ch_dec[i]
            num_ch_out = self.num_ch_enc[i]

            if self.concat_skips:
                num_ch_in += self.num_ch_enc[i]

            self.convs[f"upconv_{i}"] = ConvBlock(num_ch_in, num_ch_out, activation)

            num_ch_in = self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i - 1] if i != 0 else self.num_ch_dec[i]
            self.convs[f"unpack_{i}"] = UnpackLayerConv3d(num_ch_in, num_ch_out, 3, d=num_3d_feat)

        self.convs["last_conv"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0]//2, activation)

        self.convs["dispconv"] = Conv3x3(self.num_ch_dec[0]//2, self.num_output_channels)
        if self.uncertainty:
            self.convs["uncertaintyconv"] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        x = self.convs["first_upconv"](x)
        x = self.convs["first_unpack"](x)

        for i in range(1, -1, -1): # [1, 0]

            if self.concat_skips:
                x = [x, input_features[i]]
                x = torch.cat(x, 1)
            else:
                x = x + input_features[i]

            x = self.convs[f"upconv_{i}"](x)
            x = self.convs[f"unpack_{i}"](x)

        x = self.convs["last_conv"](x)

        self.outputs["disp"] = self.sigmoid(self.convs["dispconv"](x))

        if self.uncertainty:
            # predicting log(var(disp)) (unbounded) instead of standard deviation for stability
            self.outputs["uncertainty"] = self.convs["uncertaintyconv"](x)

        return self.outputs