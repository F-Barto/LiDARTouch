import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class BlockConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, activation_cls=None, dilation=1, norm_layer=None):
        super(BlockConv, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=dilation*(kernel_size//2),
                               dilation=dilation, bias=False)
        self.bn1 = norm_layer(out_planes)

        if activation_cls is None:
            activation_cls = nn.ReLU

        self.activation = activation_cls(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        return x

def packing(x, r=2):
    """
    Takes a [B,C,H,W] tensor and returns a [B,(r^2)C,H/r,W/r] tensor, by concatenating
    neighbor spatial pixels as extra channels. It is the inverse of nn.PixelShuffle
    (if you apply both sequentially you should get the same tensor)
    Parameters
    ----------
    x : torch.Tensor [B,C,H,W]
        Input tensor
    r : int
        Packing ratio
    Returns
    -------
    out : torch.Tensor [B,(r^2)C,H/r,W/r]
        Packed tensor
    """
    b, c, h, w = x.shape
    out_channel = c * (r ** 2)
    out_h, out_w = h // r, w // r
    x = x.contiguous().view(b, c, out_h, r, out_w, r)
    return x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)

########################################################################################################################

class PackLayerConv2d(nn.Module):
    """
    Packing layer with 2d convolutions. Takes a [B,C,H,W] tensor, packs it
    into [B,(r^2)C,H/r,W/r] and then convolves it to produce [B,C,H/r,W/r].
    """
    def __init__(self, in_channels, kernel_size, r=2, **kwargs):
        """
        Initializes a PackLayerConv2d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = BlockConv(in_channels * (r ** 2), in_channels, kernel_size, **kwargs)
        self.pack = partial(packing, r=r)

    def forward(self, x):
        """Runs the PackLayerConv2d layer."""
        x = self.pack(x)
        x = self.conv(x)
        return x


class UnpackLayerConv2d(nn.Module):
    """
    Unpacking layer with 2d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, **kwargs):
        """
        Initializes a UnpackLayerConv2d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        """
        super().__init__()
        self.conv = BlockConv(in_channels, out_channels * (r ** 2), kernel_size, **kwargs)
        self.unpack = nn.PixelShuffle(r)

    def forward(self, x):
        """Runs the UnpackLayerConv2d layer."""
        x = self.conv(x)
        x = self.unpack(x)
        return x

########################################################################################################################

class PackLayerConv3d(nn.Module):
    """
    Packing layer with 3d convolutions. Takes a [B,C,H,W] tensor, packs it
    into [B,(r^2)C,H/r,W/r] and then convolves it to produce [B,C,H/r,W/r].
    """
    def __init__(self, in_channels, kernel_size, r=2, d=8, **kwargs):
        """
        Initializes a PackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        self.conv = BlockConv(in_channels * (r ** 2) * d, in_channels, kernel_size, **kwargs)
        self.pack = partial(packing, r=r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the PackLayerConv3d layer."""
        x = self.pack(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.conv(x)
        return x


class UnpackLayerConv3d(nn.Module):
    """
    Unpacking layer with 3d convolutions. Takes a [B,C,H,W] tensor, convolves it
    to produce [B,(r^2)C,H,W] and then unpacks it to produce [B,C,rH,rW].
    """
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8, **kwargs):
        """
        Initializes a UnpackLayerConv3d object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        r : int
            Packing ratio
        d : int
            Number of 3D features
        """
        super().__init__()
        self.conv = BlockConv(in_channels, out_channels * (r ** 2) // d, kernel_size, **kwargs)
        self.unpack = nn.PixelShuffle(r)
        self.conv3d = nn.Conv3d(1, d, kernel_size=(3, 3, 3),
                                stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x):
        """Runs the UnpackLayerConv3d layer."""
        x = self.conv(x)
        x = x.unsqueeze(1)
        x = self.conv3d(x)
        b, c, d, h, w = x.shape
        x = x.view(b, c * d, h, w)
        x = self.unpack(x)
        return x