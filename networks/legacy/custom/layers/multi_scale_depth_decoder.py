# code from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/resnet/depth_decoder.py

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from networks.legacy.monodepth2.layers.common import ConvBlock, Conv3x3, nearest_upsample



class MultiScaleDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, activation, scales=range(4), num_output_channels=1,
                 concat_skips=True, uncertainty=False ):
        super(MultiScaleDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.concat_skips = concat_skips
        self.scales = scales
        self.uncertainty = uncertainty
        self.num_ch_enc = num_ch_enc # [64, 64, 128]
        self.num_ch_dec = np.array([16,32,64,128])


        # decoder
        self.convs = OrderedDict()

        self.convs["first_upconv"] = ConvBlock(self.num_ch_enc[-1], self.num_ch_dec[-1], activation)

        for i in range(2, 0, -1):  # [2, 1]
            # upconv pre upsampling
            num_ch_in = self.num_ch_dec[i+1] # i= [3, 2] ; num_ch_in = [128, 64]
            num_ch_out = self.num_ch_dec[i] # i= [2, 1] ; num_ch_out = [64, 32]

            if self.concat_skips:
                num_ch_in += self.num_ch_enc[i-1] # i= [1, 0] ; num_ch_out = [64, 64]

            self.convs[f"upconv_{i}"] = ConvBlock(num_ch_in, num_ch_out, activation)

        self.convs["last_conv"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0], activation) #in 32; out 16

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            if self.uncertainty:
                self.convs[("uncertaintyconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        x = self.convs["first_upconv"](x)

        i = self.scales[-1] # 3
        self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        if self.uncertainty:
            # predicting log(var(disp)) (unbounded) instead of standard deviation for stability
            self.outputs[("uncertainty", i)] = self.convs[("uncertaintyconv", i)](x)

        x = nearest_upsample(x)

        for i in range(2, 0, -1):  # [2, 1]

            if self.concat_skips:
                x = [x, input_features[i-1]] # i: [1, 0]
                x = torch.cat(x, 1)
            else:
                x = x + input_features[i-1]

            x = self.convs[f"upconv_{i}"](x)

            self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
            if self.uncertainty:
                # predicting log(var(disp)) (unbounded) instead of standard deviation for stability
                self.outputs[("uncertainty", i)] = self.convs[("uncertaintyconv", i)](x)

            x = nearest_upsample(x)

        x = self.convs["last_conv"](x)

        i = 0
        self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        if self.uncertainty:
            # predicting log(var(disp)) (unbounded) instead of standard deviation for stability
            self.outputs[("uncertainty", i)] = self.convs[("uncertaintyconv", i)](x)

        return self.outputs