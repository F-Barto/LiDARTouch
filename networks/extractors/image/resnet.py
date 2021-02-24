from torch import nn
import numpy as np

from networks.common.resnet_base import ResNetBase
from networks.common.basic_blocks import conv7x7


class ResNetExtractor(ResNetBase):

    def __init__(self, block, layers, activation, zero_init_residual=False, input_channels=3, **kwargs):
        super().__init__(**kwargs)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.inplanes = 64

        ############### first conv ###############
        self.conv1 = conv7x7(input_channels, self.inplanes, stride=2, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)

        self.activation = activation(inplace=True)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ############### body ###############
        self.layer1 = self._make_layer(block, 64, layers[0], activation)
        self.layer2 = self._make_layer(block, 128, layers[1], activation, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], activation, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], activation, stride=2)

        self.init_weights(zero_init_residual)

    def forward(self, x):

        self.features = []
        x = self.conv1(x)
        x = self.bn1(x)
        self.features.append(self.activation(x))

        x = self.pooling(self.features[-1])
        self.features.append(self.layer1(x))

        self.features.append(self.layer2(self.features[-1]))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))

        return self.features