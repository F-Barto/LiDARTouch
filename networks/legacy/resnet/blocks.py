from torch import nn
from torch.nn.utils import spectral_norm as spectral_norm_conv
from networks.legacy.spectral_norm_fc import spectral_norm_fc

def conv7x7(in_planes, out_planes, stride=1, groups=1, bias=False, spectral_norm=False, n_power_iterations=5):
    """3x3 convolution with padding"""

    if spectral_norm:
        # use spectral norm fc, because bound are tight for 1x1 convolutions
        conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=1, padding=3, groups=groups, bias=True)
        return spectral_norm_conv(conv_layer, n_power_iterations=n_power_iterations)

    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, groups=groups, bias=bias)

def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False, dilation=1, spectral_norm=False, n_power_iterations=5):
    """3x3 convolution with padding"""

    if spectral_norm:
        # use spectral norm fc, because bound are tight for 1x1 convolutions
        conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, groups=groups, bias=True)
        return spectral_norm_conv(conv_layer, n_power_iterations=n_power_iterations)

    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False, spectral_norm=False, n_power_iterations=5):
    """1x1 convolution"""

    if spectral_norm:
        # use spectral norm fc, because bound are tight for 1x1 convolutions
        conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
        return spectral_norm_fc(conv_layer, n_power_iterations)

    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, activation_cls, stride=1, dilation=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.activation = activation_cls(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, activation_cls, stride=1, downsample=None, groups=1, dilation=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation=dilation,)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation = activation_cls(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


#######################################################################################
############################# Pre-Activated Blocks ####################################
#######################################################################################

class PreActBasicBlock(nn.Module):
    r"""pre-activated BasicBlock from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for pre-activated ResNet for 18, 34 layers.
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size.
        downsample (Module) optional downsample module to downsample the input.
    """
    expansion = 1

    def __init__(self, inplanes, planes, activation_cls, stride=1, downsample=None, norm_layer=None):
        super(PreActBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.activation = activation_cls(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

    def forward(self, x):

        out = self.bn1(x)
        out = self.activation(out)

        identity = self.downsample(out) if self.downsample is not None else x

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        return out + identity

##################################################################################################
############################# Invertible Pre-Activated Blocks ####################################
##################################################################################################

class InvertiblePreActBasicBlock(nn.Module):
    r"""pre-activated BasicBlock from
    `"Identity Mappings in Deep Residual Networks"<https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for pre-activated ResNet for 18, 34 layers.
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size.
        downsample (Module) optional downsample module to downsample the input.
    """
    expansion = 1

    def __init__(self, inplanes, planes, activation_cls,**kwargs):
        super(InvertiblePreActBasicBlock, self).__init__()

        self.activation = activation_cls(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, spectral_norm=True, **kwargs)
        self.conv2 = conv3x3(planes, planes, spectral_norm=True, **kwargs)

    def forward(self, x):

        out = self.activation(x)
        out = self.conv1(out)

        out = self.activation(out)
        out = self.conv2(out)

        return out + x