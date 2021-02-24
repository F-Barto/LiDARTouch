from torch import nn
from .blocks import conv1x1, conv7x7, BasicBlock, Bottleneck, PreActBasicBlock, InvertiblePreActBasicBlock
from networks.legacy.pixelunshuffle import PixelUnshuffle
from networks.legacy.monodepth2.layers.common import SubPixelDownsamplingBlock



class ResNet(nn.Module):

    def __init__(self, block, layers, activation, zero_init_residual=False, groups=1, width_per_group=64,
                 norm_layer=None, input_channels=3, no_first_norm=False, no_maxpool=False, preact=False, invertible=False, **kwargs):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.no_maxpool = no_maxpool
        self.no_first_norm = no_first_norm
        self.preact = preact
        self.invertible = invertible

        self.inplanes = 64

        self.groups = groups
        self.base_width = width_per_group

        ############### first conv ###############
        if self.invertible:
            self.conv1 = conv7x7(input_channels, self.inplanes // 4, stride=1, groups=1, spectral_norm=True, **kwargs)
        else:
            self.conv1 = conv7x7(input_channels, self.inplanes, stride=2, bias=self.no_first_norm)

        if not self.no_first_norm and not self.invertible:
            self.bn1 = norm_layer(self.inplanes)
        self.activation = activation(inplace=True)

        if self.invertible:
            self.pooling = PixelUnshuffle(2)
        else:
            if not self.no_maxpool:
                self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            else:
                self.pooling = nn.Sequential(
                    conv1x1(self.inplanes, self.inplanes, stride=2),
                    norm_layer(self.inplanes),
                    activation(inplace=True)
                )

        ############### body ###############
        self.layer1 = self._make_layer(block, 64, layers[0], activation, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], activation, stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], activation, stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], activation, stride=2, **kwargs)

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

    def _make_layer(self, block, planes, blocks, activation, stride=1, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion) and not self.invertible:
            if not self.preact:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []

        if self.invertible:
            if stride != 1:
                layers.append(SubPixelDownsamplingBlock(self.inplanes, out_channels=planes, downscale_factor=2,
                                                        activation=activation, spectral_norm=True, **kwargs))
            else:
                layers.append(block(self.inplanes, planes, activation, **kwargs))

            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, activation, **kwargs))

        else:
            layers.append(block(self.inplanes, planes, activation, stride, downsample=downsample, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, activation, groups=self.groups,
                                    base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        if not self.no_first_norm and not self.invertible:
            x = self.bn1(x)

        x = self.activation(x)
        x = self.pooling(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, activation, **kwargs):
    model = ResNet(block, layers, activation, **kwargs)

    return model

def resnet18(activation, preact=False, invertible=False, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    block = BasicBlock
    if preact:
        block = PreActBasicBlock
    if invertible:
        preact = True
        block = InvertiblePreActBasicBlock

    return _resnet(block, [2, 2, 2, 2], activation, preact=preact, invertible=invertible, **kwargs)


def resnet34(activation, preact=False, invertible=False, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    block = BasicBlock
    if preact:
        block = PreActBasicBlock
    if invertible:
        preact = True
        block = InvertiblePreActBasicBlock

    return _resnet(block, [3, 4, 6, 3], activation, preact=preact, invertible=invertible, **kwargs)


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