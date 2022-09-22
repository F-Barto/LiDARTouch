import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_disp(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp

    return scaled_disp

def get_activation(activation_name):
    activations = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'elu': nn.ELU,
    }

    valid_activation_names = list(activations.keys())

    if activation_name not in valid_activation_names:
        raise  NotImplementedError(f"Invalid activation function name ({activation_name}),"
                                   f" valid ones are: {valid_activation_names}")

    return activations[activation_name]



def conv7x7(in_planes, out_planes, stride=1, groups=1, bias=False):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3,  groups=groups, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def ConvBN(in_planes, out_planes, kernel_size, stride, pad, dilation):
    """
    Perform 2D Convolution with Batch Normalization
    """
    return nn.Sequential([
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size = kernel_size,
            stride = stride,
            padding = dilation if dilation > 1 else pad,
            dilation = dilation,
            bias=False
        ),
        nn.BatchNorm2d(out_planes)
    ])

class PaddedConv3x3Block(nn.Module):
    """Layer to perform a convolution followed by activation function
    """
    def __init__(self, in_channels, out_channels, activation=nn.ELU):
        super().__init__()

        self.conv = PaddedConv3x3(in_channels, out_channels)
        self.activation = activation(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out


class PaddedConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super().__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class PyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module as defined in:

     Pyramid Scene Parsing Network, Zhao et al., https://arxiv.org/pdf/1612.01105.pdf

    for original implementation see:
    https://github.com/hszhao/PSPNet/blob/master/evaluation/prototxt/pspnet50_ADE20K_473.prototxt


    the original sizes are 10, 20, 30, 60
    """

    def __init__(self, in_chans, activation_cls, pool_mode='avg', out_chans=512, sizes=(8, 16, 32, 64)):
        super().__init__()
        self.levels = []
        inner_bottleneck_chans = in_chans // len(sizes)
        self.levels = nn.ModuleList([self._make_level(in_chans, inner_bottleneck_chans, size) for size in sizes])
        self.end_bottleneck = nn.Conv2d(in_chans * 2, out_chans, kernel_size=1)
        self.activation = activation_cls(inplace=True)

        assert pool_mode in ['max', 'avg'], f"Invalid pooling mode '{pool_mode}', expected 'max' or 'avg'"

        self.pooling = {
            'max': nn.MaxPool2d,
            'avg': nn.AvgPool2d
        }[pool_mode]

    def _make_level(self, in_chans, out_chans, size):
        return nn.Sequential([
            self.pooling(kernel_size=size, stride=size, ceil_mode=True),
            nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chans),
            self.activation,
        ])

    def forward(self, input_features):
        h, w = input_features.size(2), input_features.size(3)
        pyramid = [F.interpolate(input=level(input_features), size=(h, w), mode='bilinear') for level in self.levels]
        global_context = pyramid + [input_features]
        out = self.end_bottleneck(torch.cat(global_context, 1))
        return self.activation(out)


def nearest_upsample(x, scale_factor=2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode="nearest")

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    ICNR init of `x`, with `scale` and `init` function.
    - https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

class SubPixelUpsamplingBlock(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    "useful conversation: https://twitter.com/jeremyphoward/status/1066429286771580928"
    def __init__(self, in_channels, out_channels=None, upscale_factor=2, blur=True):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor * upscale_factor), kernel_size=3, stride=1,
                               padding=1, bias=True)

        icnr(self.conv.weight)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur

    def forward(self,x):

        x = self.conv(x)
        x = self.pixel_shuffle(x)
        if self.do_blur:
            x = self.pad(x)
            x = self.blur(x)
        return x

class FSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super().__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.elu = nn.ELU(inplace=True)

    def forward(self, high_features, low_features):
        features = [nearest_upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.elu(self.conv_se(features))

