from torch import nn
from networks.legacy.resnet.blocks import conv1x1, conv7x7, BasicBlock, Bottleneck
import numpy as np

from .packing import PackLayerConv3d


# Inspired by PSPNet

class DilatedPackEncoder(nn.Module):

    def __init__(self, block, layers, activation, zero_init_residual=False, groups=1, width_per_group=64,
                 norm_layer=None, input_channels=3, dilation=True, **kwargs):
        super(DilatedPackEncoder, self).__init__()

        self.num_ch_enc = np.array([32, 64, 512])

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32

        self.groups = groups
        self.base_width = width_per_group

        ############### first conv ###############
        self.conv1 = conv7x7(input_channels, self.inplanes, stride=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.activation = activation(inplace=True)

        ################ Packing convs ###############
        pack_kernel = [5,3,3]
        num_3d_feat = 4

        dilations = [2, 4] if dilation else [1, 1]

        self.pack1 = PackLayerConv3d(32, pack_kernel[0], d=num_3d_feat)
        self.pack2 = PackLayerConv3d(64, pack_kernel[1], d=num_3d_feat)
        self.pack3 = PackLayerConv3d(128, pack_kernel[2], d=num_3d_feat)

        ############### body ###############
        self.layer1 = self._make_layer(block, 64, layers[0], activation, pack=self.pack2, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], activation, pack=self.pack3, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], activation, dilation=dilations[0], **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], activation, dilation=dilations[1], **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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


    def _make_layer(self, block, planes, blocks, activation, stride=1, dilation=1, pack=None, **kwargs):
        norm_layer = self._norm_layer

        layers = []

        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers.append(block(self.inplanes, planes, activation, stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, norm_layer=norm_layer))

        if pack is not None:
            layers.append(pack)

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation, dilation=dilation,
                                groups=self.groups, base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        self.features = []
        x = self.conv1(x)
        x = self.bn1(x)
        self.activation(x)
        self.features.append(self.pack1(x))

        self.features.append(self.layer1(self.features[-1]))

        feature = self.layer2(self.features[-1])
        feature = self.layer3(feature)
        self.features.append(self.layer4(feature))

        return self.features


def _resnet(block, layers, activation, **kwargs):
    model = DilatedPackEncoder(block, layers, activation, **kwargs)

    return model

def resnet18(activation, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    block = BasicBlock

    return _resnet(block, [2, 2, 2, 2], activation, **kwargs)


def resnet34(activation, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    block = BasicBlock

    return _resnet(block, [3, 4, 6, 3], activation, **kwargs)


def resnet50(activation, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], activation, **kwargs)


def resnet101(activation, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], activation, **kwargs)


def resnet152(activation, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], activation, **kwargs)