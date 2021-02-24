from torch import nn
import numpy as np

from networks.common.resnet_base import ResNetBase
from networks.common.basic_blocks import conv7x7, conv1x1


class DRN_C(ResNetBase):

    """
    As defined in
    https://towardsdatascience.com/review-drn-dilated-residual-networks-image-classification-semantic-segmentation-d527e1a8fb5
    https://github.com/fyu/drn/blob/master/drn.py
    """

    def __init__(self, block, layers, activation, zero_init_residual=False, input_channels=3, **kwargs):
        super().__init__(**kwargs)

        self.activation = activation(inplace=True)

        self.num_ch_enc = np.array([32, 64, 128, 512])
        self.inplanes = 16

        ############### first conv ###############

        self.layer0 = nn.Sequential(*[
            conv7x7(input_channels, self.inplanes, stride=1, bias=False),
            self._norm_layer(self.inplanes),
            activation(inplace=True),
            self._make_layer(block, 16, 1, activation),
            self._make_layer(block, 32, 1, activation, stride=2)
        ])

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

    def forward(self, x):

        self.features = []
        self.features.append(self.layer0(x))
        self.features.append(self.layer1(self.features[-1]))
        self.features.append(self.layer2(self.features[-1]))

        x = self.layer3(self.features[-1])
        x = self.layer4(x)
        x = self.layer5(x)
        self.features.append(self.layer6(x))

        return self.features