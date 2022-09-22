from torch import nn

from lidartouch.networks.layers.basic_blocks import conv1x1
from lidartouch.networks.layers.resnet_blocks import BasicBlock, Bottleneck
from functools import partial

class ResNetBase(nn.Module):

    def __init__(self, groups=1, width_per_group=64, norm_layer=None, **kwargs):
        super().__init__(**kwargs)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.groups = groups
        self.base_width = width_per_group

    def init_weights(self, zero_init_residual):
        """Initializes network weights."""

        if isinstance(self.activation, nn.ReLU):
            initializer = partial(nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')
        else:# ELU
            initializer = nn.init.xavier_uniform_

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, activation, stride=1, dilation=1):
        norm_layer = self._norm_layer
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, activation, stride, downsample=downsample,
                            groups=self.groups, base_width=self.base_width, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation, dilation=dilation,
                                groups=self.groups, base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        raise NotImplementedError


def build_network(resnet_class, activation, version=18, **kwargs):
    """
    builds a resnet
    "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>
    with as variation as defined by given resnet_class

    :param resnet_class:
    :param activation:
    :param version: must be in [18, 34, 50, 101, 152]
    :return:
    """

    assert version in [18, 34, 50, 101, 152]

    resnets = {
        18: partial(resnet_class, BasicBlock, [2, 2, 2, 2]),
        34: partial(resnet_class, BasicBlock, [3, 4, 6, 3]),
        50: partial(resnet_class, Bottleneck, [3, 4, 6, 3]),
        101: partial(resnet_class, Bottleneck, [3, 4, 23, 3]),
        152: partial(resnet_class, Bottleneck, [3, 8, 36, 3]),
    }

    return resnets[version](activation, **kwargs)


