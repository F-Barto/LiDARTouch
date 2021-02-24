# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/resnet/resnet_encoder.py
# and https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py

from __future__ import absolute_import, division, print_function

import numpy as np

import torch.nn as nn
from networks.legacy.resnet import resnet as models
from networks.legacy.resnet.blocks import BasicBlock, Bottleneck
# import torchvision.models as models


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, activation, num_input_images=1, input_channels=3):
        super(ResNetMultiImageInput, self).__init__(block, layers, activation, input_channels=input_channels,
                                                    no_first_norm=False, invertible=False)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if not self.no_first_norm:
            self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation)
        self.layer2 = self._make_layer(block, 128, layers[1], activation, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], activation, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], activation, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, activation, num_input_images=1, input_channels=3):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 34, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
    }[num_layers]

    block_type = {
        18: BasicBlock,
        34: BasicBlock,
        50: Bottleneck
    }[num_layers]

    model = ResNetMultiImageInput(block_type, blocks, activation, num_input_images=num_input_images,
                                  input_channels=input_channels)

    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, activation, num_input_images=1, input_channels=3, no_first_norm=False,
                 invertible=False, **kwargs):
        super(ResnetEncoder, self).__init__()

        if invertible:
            self.num_ch_enc = np.array([64//4, 64, 128, 256, 512])
        else:
            self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, activation, num_input_images,
                                                   input_channels=input_channels)
        else:
            self.encoder = resnets[num_layers](activation, input_channels=input_channels,
                                               no_first_norm=no_first_norm, invertible=invertible, **kwargs)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = self.encoder.conv1(input_image)
        if not self.encoder.no_first_norm and not self.encoder.invertible:
            x = self.encoder.bn1(x)
        self.features.append(self.encoder.activation(x))
        self.features.append(self.encoder.layer1(self.encoder.pooling(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features