from torch import nn
from networks.legacy.resnet.blocks import conv1x1, conv7x7, BasicBlock, Bottleneck
import numpy as np
import torch.nn.functional as F
import torch

# Inspired by PSPNet

class DilatedResNetEncoder(nn.Module):

    def __init__(self, block, layers, activation, zero_init_residual=False, groups=1, width_per_group=64,
                 norm_layer=None, input_channels=3, no_maxpool=False, dilation=True, strided=False, small=True,
                 **kwargs):
        super(DilatedResNetEncoder, self).__init__()

        self.strided = strided

        self.small = small

        self.num_ch_enc = np.array([64, 64, 128]) if self.strided else np.array([64, 64, 512])

        if not self.small:
            self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.no_maxpool = no_maxpool

        self.inplanes = 64

        self.groups = groups
        self.base_width = width_per_group

        ############### first conv ###############
        self.conv1 = conv7x7(input_channels, self.inplanes, stride=2, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.activation = activation(inplace=True)

        if not self.no_maxpool:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1
        else:
            stride = 2

        if self.strided and self.small:
            self.up_last_group = nn.Sequential(
                nn.Conv2d(640, 128, kernel_size=3, padding=1, stride=1, bias=True),
                activation(inplace=True)
            )

        strides = [2,2] if self.strided else [1,1]

        dilations =  [2,4] if dilation else [1,1]

        ############### body ###############
        self.layer1 = self._make_layer(block, 64, layers[0], activation, stride=stride, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], activation, stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], activation, stride=strides[0],
                                       dilation=dilations[0], **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], activation, stride=strides[1],
                                       dilation=dilations[1], **kwargs)

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


    def _make_layer(self, block, planes, blocks, activation, stride=1, dilation=1, **kwargs):
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

        self.features = []
        x = self.conv1(x)
        x = self.bn1(x)
        self.features.append(self.activation(x))

        if not self.no_maxpool:
            x = self.pooling(self.features[-1])
        else:
            x = self.features[-1]

        self.features.append(self.layer1(x))

        if self.strided:
            if self.small:
                x_1_8 = self.layer2(self.features[-1])
                x_1_32 = self.layer4(self.layer3(x_1_8))
                upped_x_1_32 = F.interpolate(x_1_32, scale_factor=4, mode="nearest")

                concat = [x_1_8, upped_x_1_32]
                concat = torch.cat(concat, 1)

                feature = self.up_last_group(concat)

                self.features.append(feature)
            else:
                self.features.append(self.layer2(self.features[-1]))
                self.features.append(self.layer3(self.features[-1]))
                self.features.append(self.layer4(self.features[-1]))
        else:
            feature = self.layer2(self.features[-1])
            feature = self.layer3(feature)
            self.features.append(self.layer4(feature))

        return self.features


def _resnet(block, layers, activation, **kwargs):
    model = DilatedResNetEncoder(block, layers, activation, **kwargs)

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