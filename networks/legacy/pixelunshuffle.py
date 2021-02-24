import torch
from torch import Tensor
from torch import nn

def pixel_unshuffle(x: torch.Tensor, downscale_factor: int):
    downscale_factor_sqr = downscale_factor * downscale_factor
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC*(downscale_factor_sqr), iH//downscale_factor, iW//downscale_factor
    y = y.reshape(B, iC, oH, downscale_factor, oW, downscale_factor)
    y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y

class PixelUnshuffle(nn.Module):
    r"""Rearranges elements in a tensor of shape :math:`(*, C, H \times r, W \times r)`
    to a tensor of shape :math:`(*, C \times r^2, H, W)`.
    This is the inverse of the PixelShuffle operation from the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.
    Args:
        downscale_factor (int): factor to increase spatial resolution by
    """

    def __init__(self, downscale_factor: int) -> None:
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return pixel_unshuffle(input, self.downscale_factor)