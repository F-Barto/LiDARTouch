# code from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/resnet/layers.py

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/layers.py

from __future__ import absolute_import, division, print_function
from networks.legacy.resnet.blocks import conv3x3

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mish import MishAuto

from networks.legacy.pixelunshuffle import PixelUnshuffle

def get_activation(activation_name):
    activations = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'elu': nn.ELU,
        'mish': MishAuto
    }

    possible_activation_names = list(activations.keys())

    assert activation_name in possible_activation_names, f"Activation func name must be in {possible_activation_names}"

    return activations[activation_name]




def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by activation function
    """
    def __init__(self, in_channels, out_channels, activation):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.activation = activation(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def nearest_upsample(x, scale_factor=2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode="nearest")

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    ICNR init of `x`, with `scale` and `init` function.
    - https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class SubPixelUpsamplingBlock(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    "useful conversation: https://twitter.com/jeremyphoward/status/1066429286771580928"
    def __init__(self, in_channels, out_channels=None, upscale_factor=2, blur=True):
        super(SubPixelUpsamplingBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor * upscale_factor), kernel_size=3, stride=1,
                               padding=1, bias=True)

        icnr(self.conv.weight)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur

    def forward(self,x):

        x = self.conv(x)
        x = self.pixel_shuffle(x)
        if self.do_blur:
            x = self.pad(x)
            x = self.blur(x)
        return x

class SubPixelDownsamplingBlock(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    "useful conversation: https://twitter.com/jeremyphoward/status/1066429286771580928"
    def __init__(self, in_channels, out_channels=None, downscale_factor=2, activation=None,
                 spectral_norm=True, n_power_iterations=5):

        super(SubPixelDownsamplingBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels


        self.activation = activation(inplace=True) if activation is not None else None

        self.conv = conv3x3(in_channels, out_channels // (downscale_factor * downscale_factor), stride=1, bias=True,
                            spectral_norm=spectral_norm, n_power_iterations =n_power_iterations)
        self.pixel_unshuffle = PixelUnshuffle(downscale_factor)

    def forward(self,x):

        x = self.conv(x)
        x = self.pixel_unshuffle(x)

        if self.activation is not None:
            x = self.activation(x)

        return x