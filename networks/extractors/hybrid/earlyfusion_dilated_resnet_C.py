import torch
from torch import nn
import numpy as np

from networks.common.resnet_base import ResNetBase
from networks.common.basic_blocks import conv7x7, conv1x1, conv3x3
from networks.common.fusion.projector import Projector
from utils.camera import Camera


class DRN_C(ResNetBase):

    """
    As defined in
    https://towardsdatascience.com/review-drn-dilated-residual-networks-image-classification-semantic-segmentation-d527e1a8fb5
    https://github.com/fyu/drn/blob/master/drn.py
    """

    def __init__(self, block, layers, activation, zero_init_residual=False, input_channels=3, **kwargs):
        super().__init__(**kwargs)

        self.activation = activation(inplace=True)

        self.num_ch_enc = np.array([32, 64, 512])
        self.inplanes = 16

        ############### first conv ###############

        self.layer0 = nn.Sequential(*[
            conv7x7(input_channels, self.inplanes, stride=1, bias=False),
            self._norm_layer(self.inplanes),
            activation(inplace=True),
            self._make_layer(block, 16, 1, activation),
            self._make_layer(block, 32, 1, activation, stride=2)
        ])

        # early fusion blocks
        self.img_fusion_mapping = conv1x1(32, 32, stride=1, bias=False)
        self.projector = Projector(self.num_ch_enc[-1], 32, learn_zero_encoding=True, linear_mapping=True)
        self.post_concat = nn.Sequential(
            conv1x1(32*2, 32, bias=False),
            self._norm_layer(32),
            activation(inplace=True),
        )

        ############### body ###############
        self.layer1 = self._make_layer(block, 64, layers[0], activation, stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], activation, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], activation, stride=1, dilation=2, new_level=False)
        self.layer4 = self._make_layer(block, 512, layers[2], activation, stride=1, dilation=4, new_level=False)

        # These 2 blocks have progressively lowered dilation to remove the aliasing artifacts.
        # The residual connections are also removed so that artifacts can't be passed through residual connections.
        self.layer5 = self._make_layer(block, 512, 2, activation, dilation=2, new_level=False, residual=False)
        self.layer6 = self._make_layer(block, 512, 2, activation, dilation=1, new_level=False, residual=False)

        self.init_weights(zero_init_residual)

    def _make_layer(self, block, planes, blocks, activation, stride=1, dilation=1, new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        norm_layer = self._norm_layer

        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        first_block_dilation = 1 if dilation == 1 else (dilation // 2 if new_level else dilation)
        layers.append(block(self.inplanes, planes, activation, stride=stride, downsample=downsample,
                            dilation=first_block_dilation, residual=residual))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation, residual=residual, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, image_input, pc_input, intrinsics):

        B, _, H, W = image_input.shape


        self.features = []

        image_input = self.layer0(image_input)
        image_fusion_input = self.img_fusion_mapping(image_input)

        # project 3D-to-2D
        _, _, downsampled_H, downsampled_W = image_input.shape
        scale_factor = downsampled_W / float(W)
        cam = Camera(K=intrinsics.float()).scaled(scale_factor).to(image_input.device)
        lidar_features = self.projector(cam, pc_input, B, downsampled_H, downsampled_W)

        # fuse
        x = torch.cat([image_fusion_input,lidar_features], 1)
        x = self.post_concat(x)
        #x = image_input + x
        self.features.append(x)

        self.features.append(self.layer1(self.features[-1]))

        x = self.layer2(self.features[-1])
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        self.features.append(self.layer6(x))

        return self.features