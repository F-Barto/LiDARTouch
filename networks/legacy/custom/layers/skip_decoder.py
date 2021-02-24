# code from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/resnet/depth_decoder.py


from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from networks.legacy.monodepth2.layers.common import ConvBlock, Conv3x3, nearest_upsample, SubPixelUpsamplingBlock


class Conv1x1Block(nn.Module):
    """Layer to perform a convolution followed by activation function
    """
    def __init__(self, in_channels, out_channels, activation):
        super(Conv1x1Block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.activation = activation(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out

class Conv3x3Block(nn.Module):
    """Layer to perform a convolution followed by activation function
    """
    def __init__(self, in_channels, out_channels, activation):
        super(Conv3x3Block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.activation = activation(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out

class SkipDecoder(nn.Module):
    def __init__(self, num_ch_enc, activation, scales=3, num_output_channels=1, upsample_path='direct',
                 upsample_mode='nearest', blur=True, blur_at_end=True, refinement=False):
        super(SkipDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        available_upmodes = ['nearest', 'pixelshuffle', 'res-pixelshuffle']
        if upsample_mode not in available_upmodes:
            raise ValueError(f"upsample_mode must be in {available_upmodes} | upsample_mode={upsample_mode}")
        self.upsample_mode = upsample_mode

        available_uppaths = ['direct', 'cascaded', 'conv1cascaded']
        if upsample_path not in available_uppaths:
            raise ValueError(f"upsample_path must be in {available_uppaths} | upsample_path={upsample_path}")
        self.upsample_path = upsample_path

        self.num_ch_enc = num_ch_enc
        self.num_ch_concat = sum(num_ch_enc)

        self.upscale_factors = np.zeros(len(num_ch_enc)).astype(int)
        if self.upsample_path == 'direct':
            self.upscale_factors = [2,4,8]
        else:
            self.upscale_factors = [2]*3
        print(f"SkipDecoder upscale_factors={self.upscale_factors}")


        # decoder
        self.convs = OrderedDict()
        for i in range(self.scales-1, -1, -1): # [2, 1, 0]
            # upconv_0, pre upsampling
            num_ch_in = self.num_ch_enc[i]
            if 'pixelshuffle' in self.upsample_mode and self.upsample_path != 'direct':
                do_blur = blur and (i != 0 or blur_at_end)
                self.convs[("pixelshuffle", i)] = SubPixelUpsamplingBlock(num_ch_in, blur=do_blur,
                                                                          upscale_factor=self.upscale_factors[i])
            num_ch_in = self.num_ch_enc[i-1]
            if self.upsample_path == 'conv1cascaded':
                self.convs[("skipconv", i-1)] = Conv1x1Block(num_ch_in, num_ch_in, activation)

        # reduce to 128 for pixelshffle -> 128 * 4^2 = 2048 or else 256 * 4^2 = 4096 too big
        concat_chans_out = 128 if 'pixelshuffle' in self.upsample_mode else 256
        self.concatconv = ConvBlock(self.num_ch_concat, concat_chans_out, activation)

        self.dispconv = Conv3x3(concat_chans_out, self.num_output_channels)

        if 'pixelshuffle' in self.upsample_mode:
            self.last_pixelshuffle = SubPixelUpsamplingBlock(concat_chans_out, out_channels=self.num_output_channels,
                                                             blur=blur_at_end, upscale_factor=4)

        self.refinement = refinement

        if self.refinement:
            self.refine_block = nn.Sequential(
                Conv3x3Block(self.num_output_channels + concat_chans_out, 64, activation),
                Conv3x3Block(64, 32, activation),
                Conv3x3(32, self.num_output_channels, activation)
        )

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def upsample(self, x, i):

        if self.upsample_mode == 'nearest' or self.upsample_path == 'direct':
            x = nearest_upsample(x, scale_factor=self.upscale_factors[i])
        elif self.upsample_mode == 'pixelshuffle':
            x = self.convs[("pixelshuffle", i)](x)
        elif self.upsample_mode == 'res-pixelshuffle':
            x = self.convs[("pixelshuffle", i)](x) + nearest_upsample(x, scale_factor=self.upscale_factors[i])
        else:
            pass

        return x


    def forward(self, input_features):

        concat = self.upsample(input_features[-1], self.scales-1)

        for i in range(self.scales-2, -1, -1): # [1, 0]

            if self.upsample_path == 'conv1cascaded':
                skip = self.convs[("skipconv", i)](input_features[i])
            else:
                skip = input_features[i]

            if self.upsample_path == 'direct':
                skip = self.upsample(skip, i)
                concat = [concat, skip]
                concat = torch.cat(concat, 1)
            else: # cascaded
                concat = [concat, skip]
                concat = torch.cat(concat, 1)
                concat = self.upsample(concat, i)

        x = self.concatconv(concat)

        """
        if 'pixelshuffle' in self.upsample_mode:
            up_x = self.last_pixelshuffle(x)
        else:
            out_dispconv = self.dispconv(x)
            up_x = nearest_upsample(out_dispconv, scale_factor=2)
            
        disp = self.sigmoid(up_x)
        """

        disp_x = self.dispconv(x)
        disp = self.sigmoid(disp_x)

        if self.refinement:
            concat = [x, disp]
            concat = torch.cat(concat, 1)
            refined_disp = self.sigmoid(self.refine_block(concat))
            return {
                'coarse_disp': disp,
                'disp': refined_disp
            }

        return {'disp': disp}