# code from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/resnet/depth_decoder.py

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .common import ConvBlock, Conv3x3, nearest_upsample, SubPixelUpsamplingBlock


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, activation, scales=range(4), num_output_channels=1, use_skips=True,
                 concat_skips=True, upsample_mode='nearest', blur=True, blur_at_end=True, uncertainty=False,
                 refine_preds=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.concat_skips = concat_skips
        self.scales = scales
        self.uncertainty = uncertainty
        self.refine_preds = refine_preds

        available_upmodes = ['nearest', 'pixelshuffle', 'res-pixelshuffle']
        if upsample_mode not in available_upmodes:
            raise ValueError(f"upsample_mode must be in ['nearest', 'pixelshuffle'] | upsample_mode={upsample_mode}")
        self.upsample_mode = upsample_mode



        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16,32,64,128,256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0, pre upsampling
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_in += 1 if self.refine_preds and i < 3 else 0
            num_ch_in += 1 if self.refine_preds and i < 3  and self.uncertainty else 0
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, activation)

            if 'pixelshuffle' in self.upsample_mode:
                do_blur = blur and (i != 0 or blur_at_end)
                self.convs[("pixelshuffle", i)] = SubPixelUpsamplingBlock(num_ch_out, upscale_factor=2, blur=do_blur)

            # upconv_1, post upsampling
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0 and self.concat_skips:
                num_ch_in += self.num_ch_enc[i - 1]

            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, activation)

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
        for i in range(4, -1, -1):

            if self.refine_preds and i < 3:
                x = [x, self.outputs[("disp", i+1)]]
                if self.uncertainty:
                    x.append(self.outputs[("uncertainty", i+1)])
                x = torch.cat(x, 1)

            x = self.convs[("upconv", i, 0)](x)

            if self.upsample_mode == 'pixelshuffle':
                x = self.convs[("pixelshuffle", i)](x)
            if self.upsample_mode == 'res-pixelshuffle':
                x = self.convs[("pixelshuffle", i)](x) + nearest_upsample(x)
            if self.upsample_mode == 'nearest':
                x = nearest_upsample(x)

            if self.use_skips and i > 0:
                if self.concat_skips:
                    x = [x, input_features[i - 1]]
                    x = torch.cat(x, 1)
                else:
                    x = x + input_features[i - 1]

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

                if self.uncertainty:
                    # predicting log(var(disp)) (unbounded) instead of standard deviation for stability
                    self.outputs[("uncertainty", i)] = self.convs[("uncertaintyconv", i)](x)

        return self.outputs