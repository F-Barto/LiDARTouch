# Adapted from Packnet-sfm
# https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/pose/PoseNet.py
# Which itself adapted from SfmLearner
# https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/models/PoseExpNet.py

import torch
import torch.nn as nn

from utils.mish import MishAuto

########################################################################################################################

def conv_gn(in_planes, out_planes, kernel_size=3, activation_fn=nn.ReLU):
    """
    Convolutional block with GroupNorm
    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    kernel_size : int
        Convolutional kernel size
    Returns
    -------
    layers : nn.Sequential
        Sequence of Conv2D + GroupNorm + ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  padding=(kernel_size - 1) // 2, stride=2),
        nn.GroupNorm(16, out_planes),
        activation_fn(inplace=True)
    )

########################################################################################################################

class PoseNet(nn.Module):
    """Pose network """

    def __init__(self, nb_ref_imgs=2, input_channels = 3, activation_fn='relu'):
        super().__init__()

        assert activation_fn in ['relu', 'mish']

        activation_fn = {'relu': nn.ReLU, 'mish': MishAuto}[activation_fn]

        self.nb_ref_imgs = nb_ref_imgs

        conv_channels = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv_gn(input_channels * (1 + self.nb_ref_imgs),
                             conv_channels[0], kernel_size=7, activation_fn=activation_fn)
        self.conv2 = conv_gn(conv_channels[0], conv_channels[1], kernel_size=5, activation_fn=activation_fn)
        self.conv3 = conv_gn(conv_channels[1], conv_channels[2], activation_fn=activation_fn)
        self.conv4 = conv_gn(conv_channels[2], conv_channels[3], activation_fn=activation_fn)
        self.conv5 = conv_gn(conv_channels[3], conv_channels[4], activation_fn=activation_fn)
        self.conv6 = conv_gn(conv_channels[4], conv_channels[5], activation_fn=activation_fn)
        self.conv7 = conv_gn(conv_channels[5], conv_channels[6], activation_fn=activation_fn)

        self.pose_pred = nn.Conv2d(conv_channels[6], 6 * self.nb_ref_imgs,
                                   kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image, context):
        assert (len(context) == self.nb_ref_imgs)
        input = [image]
        input.extend(context)
        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        return pose