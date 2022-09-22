"""
Multi-scale Fusion and Multi-scale Prediction w/ same nb of resolutions levels from each modality
"""

from functools import partial
import torch.nn.functional as F

import torch
import torch.nn as nn

from lidartouch.networks.net_base import NetworkBase

from lidartouch.utils.identifiers import IMAGE, SPARSE_DEPTH
from lidartouch.utils.cli import DEPTH_NET


############################################# MinkowskiEngine ########################################################

import MinkowskiEngine as ME



def sparsify_features(x):
    """
    Sparsify features
    Parameters
    ----------
    x : Dense feature map [B,C,H,W]
    Returns
    -------
    Sparse feature map (features only in valid coordinates)
    """
    b, c, h, w = x.shape

    u = torch.arange(w).reshape(1, w).repeat([h, 1])
    v = torch.arange(h).reshape(h, 1).repeat([1, w])
    uv = torch.stack([v, u], 2).reshape(-1, 2)

    coords = [uv] * b
    feats = [feats.permute(1, 2, 0).reshape(-1, c) for feats in x]
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)
    return ME.SparseTensor(coordinates=coords, features=feats, device=x.device)


def sparsify_depth(x):
    """
    Sparsify depth map
    Parameters
    ----------
    x : Dense depth map [B,1,H,W]
    Returns
    -------
    Sparse depth map (range values only in valid pixels)
    """
    b, c, h, w = x.shape

    u = torch.arange(w, device=x.device).reshape(1, w).repeat([h, 1])
    v = torch.arange(h, device=x.device).reshape(h, 1).repeat([1, w])
    uv = torch.stack([v, u], 2)

    idxs = [(d > 0)[0] for d in x]

    coords = [uv[idx] for idx in idxs]
    feats = [feats.permute(1, 2, 0)[idx] for idx, feats in zip(idxs, x)]
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)
    return ME.SparseTensor(coordinates=coords, features=feats, device=x.device)


def densify_features(x, shape):
    """
    Densify features from a sparse tensor
    Parameters
    ----------
    x : Sparse tensor
    shape : Dense shape [B,C,H,W]
    Returns
    -------
    Dense tensor containing sparse information
    """
    stride = x.tensor_stride
    coords, feats = x.C.long(), x.F
    shape = (shape[0], shape[2] // stride[0], shape[3] // stride[1], feats.shape[1])
    dense = torch.zeros(shape, device=x.device)
    dense[coords[:, 0],
          coords[:, 1] // stride[0],
          coords[:, 2] // stride[1]] = feats
    return dense.permute(0, 3, 1, 2).contiguous()


def densify_add_features_unc(x, s, u, shape):
    """
    Densify and add features considering uncertainty
    Parameters
    ----------
    x : Dense tensor [B,C,H,W]
    s : Sparse tensor
    u : Sparse tensor with uncertainty
    shape : Dense tensor shape
    Returns
    -------
    Densified sparse tensor with added uncertainty
    """
    stride = s.tensor_stride
    coords, feats = s.C.long(), s.F
    shape = (shape[0], shape[2] // stride[0], shape[3] // stride[1], feats.shape[1])

    dense = torch.zeros(shape, device=s.device)
    dense[coords[:, -1],
          coords[:, 0] // stride[0],
          coords[:, 1] // stride[1]] = feats
    dense = dense.permute(0, 3, 1, 2).contiguous()

    mult = torch.ones(shape, device=s.device)
    mult[coords[:, -1],
         coords[:, 0] // stride[0],
         coords[:, 1] // stride[1]] = 1.0 - u.F
    mult = mult.permute(0, 3, 1, 2).contiguous()

    return x * mult + dense


def map_add_features(x, s):
    """
    Map dense features to sparse tensor and add them.
    Parameters
    ----------
    x : Dense tensor [B,C,H,W]
    s : Sparse tensor
    Returns
    -------
    Sparse tensor with added dense information in valid areas
    """
    stride = s.tensor_stride
    coords = s.coords.long()
    feats = x.permute(0, 2, 3, 1)
    feats = feats[coords[:, -1],
                  coords[:, 0] // stride[0],
                  coords[:, 1] // stride[1]]
    return ME.SparseTensor(coords=coords, feats=feats + s.feats,
                           coords_manager=s.coords_man, force_creation=True,
                           tensor_stride=s.tensor_stride)



class MinkConv2D(nn.Module):
    """
    Minkowski Convolutional Block
    Parameters
    ----------
    in_planes : number of input channels
    out_planes : number of output channels
    kernel_size : convolutional kernel size
    stride : convolutional stride
    with_uncertainty : with uncertainty or now
    add_rgb : add RGB information as channels
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride,
                 with_uncertainty=False, add_rgb=False):
        super().__init__()
        self.layer3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes * 2, kernel_size=kernel_size, stride=1, dimension=2),
            ME.MinkowskiBatchNorm(out_planes * 2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                out_planes * 2, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_planes, out_planes, kernel_size=kernel_size, stride=1, dimension=2),
        )

        self.layer_final = nn.Sequential(
            ME.MinkowskiBatchNorm(out_planes),
            ME.MinkowskiReLU(inplace=True)
        )
        self.pool = None if stride == 1 else ME.MinkowskiMaxPooling(3, stride, dimension=2)

        self.add_rgb = add_rgb
        self.with_uncertainty = with_uncertainty
        if with_uncertainty:
            self.unc_layer = nn.Sequential(
                ME.MinkowskiConvolution(
                    out_planes, 1, kernel_size=3, stride=1, dimension=2),
                ME.MinkowskiSigmoid()
            )

    def forward(self, x):
        """
        Processes sparse information
        Parameters
        ----------
        x : Sparse tensor
        Returns
        -------
        Processed tensor
        """
        if self.pool is not None:
            x = self.pool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        return None, self.layer_final(x1 + x2 + x3)


class MinkowskiEncoder(nn.Module):
    """
    Depth completion Minkowski Encoder
    Parameters
    ----------
    channels : number of channels
    with_uncertainty : with uncertainty or not
    add_rgb : add RGB information to depth features or not
    """
    def __init__(self, channels, with_uncertainty=False, add_rgb=False):
        super().__init__()
        self.mconvs = nn.ModuleList()
        kernel_sizes = [5, 5] + [3] * (len(channels) - 1)
        self.mconvs.append(
            MinkConv2D(1, channels[0], kernel_sizes[0], 2,
                       with_uncertainty=with_uncertainty))
        for i in range(0, len(channels) - 1):
            self.mconvs.append(
                MinkConv2D(channels[i], channels[i+1], kernel_sizes[i+1], 2,
                           with_uncertainty=with_uncertainty))
        self.d = self.n = self.shape = 0
        self.with_uncertainty = with_uncertainty
        self.add_rgb = add_rgb

    def prep(self, d):
        self.d = sparsify_depth(d)
        self.shape = d.shape
        self.n = 0

    def forward(self, x=None):

        unc, self.d = self.mconvs[self.n](self.d)
        self.n += 1

        if self.with_uncertainty:
            out = densify_add_features_unc(x, unc * self.d, unc, self.shape)
        else:
            out = densify_features(self.d, self.shape)

        if self.add_rgb:
            self.d = map_add_features(x, self.d)

        return out

########################################################################################################################

class Conv2D(nn.Module):
    """
    2D convolution with GroupNorm and ELU
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size
    stride : int
        Stride
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.pad = nn.ConstantPad2d([kernel_size // 2] * 4, value=0)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        """Runs the Conv2D layer."""
        x = self.conv_base(self.pad(x))
        return self.activ(self.normalize(x))


class ResidualConv(nn.Module):
    """2D Convolutional residual block with GroupNorm and ELU"""
    def __init__(self, in_channels, out_channels, stride, dropout=None):
        """
        Initializes a ResidualConv object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        stride : int
            Stride
        dropout : float
            Dropout value
        """
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride)
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.normalize = torch.nn.GroupNorm(16, out_channels)
        self.activ = nn.ELU(inplace=True)

        if dropout:
            self.conv3 = nn.Sequential(self.conv3, nn.Dropout2d(dropout))

    def forward(self, x):
        """Runs the ResidualConv layer."""
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        shortcut = self.conv3(x)
        return self.activ(self.normalize(x_out + shortcut))


def ResidualBlock(in_channels, out_channels, num_blocks, stride, dropout=None):
    """
    Returns a ResidualBlock with various ResidualConv layers.
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    num_blocks : int
        Number of residual blocks
    stride : int
        Stride
    dropout : float
        Dropout value
    """
    layers = [ResidualConv(in_channels, out_channels, stride, dropout=dropout)]
    for i in range(1, num_blocks):
        layers.append(ResidualConv(out_channels, out_channels, 1, dropout=dropout))
    return nn.Sequential(*layers)


class InvDepth(nn.Module):
    """Inverse depth layer"""
    def __init__(self, in_channels, out_channels=1, min_depth=0.5):
        """
        Initializes an InvDepth object.
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        min_depth : float
            Minimum depth value to calculate
        """
        super().__init__()
        self.min_depth = min_depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.pad = nn.ConstantPad2d([1] * 4, value=0)
        self.activ = nn.Sigmoid()

    def forward(self, x):
        """Runs the InvDepth layer."""
        x = self.conv1(self.pad(x))
        return self.activ(x) / self.min_depth

########################################################################################################################

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
    def __init__(self, in_channels, kernel_size, r=2):
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
        self.conv = Conv2D(in_channels * (r ** 2), in_channels, kernel_size, 1)
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
    def __init__(self, in_channels, out_channels, kernel_size, r=2):
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
        self.conv = Conv2D(in_channels, out_channels * (r ** 2), kernel_size, 1)
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
    def __init__(self, in_channels, kernel_size, r=2, d=8):
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
        self.conv = Conv2D(in_channels * (r ** 2) * d, in_channels, kernel_size, 1)
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
    def __init__(self, in_channels, out_channels, kernel_size, r=2, d=8):
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
        self.conv = Conv2D(in_channels, out_channels * (r ** 2) // d, kernel_size, 1)
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


########################################################################################################################

class Encoder(nn.Module):
    def __init__(self, version, in_channels, ni, n1, n2, n3, n4, n5,
                 pack_kernel, num_blocks, num_3d_feat, dropout):
        super().__init__()
        # Encoder
        self.version = version

        self.pre_calc = Conv2D(in_channels, ni, 5, 1)

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0], d=num_3d_feat)
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1], d=num_3d_feat)
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2], d=num_3d_feat)
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3], d=num_3d_feat)
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4], d=num_3d_feat)

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout)

    def forward(self, rgb):

        x = self.pre_calc(rgb)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips
        return x5p, [x, x1p, x2p, x3p, x4p]


class Decoder(nn.Module):
    def __init__(self, version, out_channels, ni, n1, n2, n3, n4, n5,
                 unpack_kernel, iconv_kernel, num_3d_feat):
        super().__init__()
        # Decoder
        self.version = version

        n1o, n1i = n1, n1 + ni + out_channels
        n2o, n2i = n2, n2 + n1 + out_channels
        n3o, n3i = n3, n3 + n2 + out_channels
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0], d=num_3d_feat)
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1], d=num_3d_feat)
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2], d=num_3d_feat)
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3], d=num_3d_feat)
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4], d=num_3d_feat)

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels)

    def forward(self, x5p, skips):
        skip1, skip2, skip3, skip4, skip5 = skips

        unpack5 = self.unpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        inv_depth4 = self.disp4_layer(iconv4)
        up_inv_depth4 = self.unpack_disp4(inv_depth4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, up_inv_depth4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, up_inv_depth4), 1)
        iconv3 = self.iconv3(concat3)
        inv_depth3 = self.disp3_layer(iconv3)
        up_inv_depth3 = self.unpack_disp3(inv_depth3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, up_inv_depth3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, up_inv_depth3), 1)
        iconv2 = self.iconv2(concat2)
        inv_depth2 = self.disp2_layer(iconv2)
        up_inv_depth2 = self.unpack_disp2(inv_depth2)

        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, up_inv_depth2), 1)
        else:
            concat1 = torch.cat((unpack1 + skip1, up_inv_depth2), 1)
        iconv1 = self.iconv1(concat1)
        inv_depth1 = self.disp1_layer(iconv1)

        if self.training:
            inv_depths = [inv_depth1, inv_depth2, inv_depth3, inv_depth4]
        else:
            inv_depths = [inv_depth1]

        return inv_depths

@DEPTH_NET
class PackNetSAN(NetworkBase):
    """
    PackNet-SAN network, from the paper (https://arxiv.org/abs/2103.16690)
    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, **kwargs):
        super().__init__()
        self.version = version[1:]
        # Input/output channels
        in_channels = 3
        out_channels = 1
        # Hyper-parameters
        ni, n1, n2, n3, n4, n5 = 32, 32, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        num_3d_feat = 4

        self.encoder = Encoder(self.version, in_channels, ni, n1, n2, n3, n4, n5,
                               pack_kernel, num_blocks, num_3d_feat, dropout)
        self.decoder = Decoder(self.version, out_channels, ni, n1, n2, n3, n4, n5,
                               unpack_kernel, iconv_kernel, num_3d_feat)

        self.mconvs = MinkowskiEncoder([n1, n2, n3, n4, n5], with_uncertainty=False)

        self.weight = torch.nn.parameter.Parameter(torch.ones(5), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(torch.zeros(5), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    @property
    def required_inputs(self):
        return [IMAGE, SPARSE_DEPTH]

    def run_network(self, rgb, input_depth=None):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x5p, skips = self.encoder(rgb)

        if input_depth is not None:
            self.mconvs.prep(input_depth)

            skips[1] = skips[1] * self.weight[0].view(1, 1, 1, 1) + self.mconvs(skips[1]) + self.bias[0].view(1, 1, 1, 1)
            skips[2] = skips[2] * self.weight[1].view(1, 1, 1, 1) + self.mconvs(skips[2]) + self.bias[1].view(1, 1, 1, 1)
            skips[3] = skips[3] * self.weight[2].view(1, 1, 1, 1) + self.mconvs(skips[3]) + self.bias[2].view(1, 1, 1, 1)
            skips[4] = skips[4] * self.weight[3].view(1, 1, 1, 1) + self.mconvs(skips[4]) + self.bias[3].view(1, 1, 1, 1)
            x5p      = x5p      * self.weight[4].view(1, 1, 1, 1) + self.mconvs(x5p)      + self.bias[4].view(1, 1, 1, 1)

        return self.decoder(x5p, skips), skips + [x5p]

    def forward(self, camera_image, sparse_depth, **kwargs):


        inv_depths, _ = self.run_network(camera_image, sparse_depth)
        return {
            'inv_depths': inv_depths,
        }
