from lidartouch.losses.reconstruction.reconstruction_loss_base import ReconstructionLossBase


import torch
import torch.nn as nn

def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMlilarity (SSIM) distance between two images.
    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters
    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim


class SSIM_L1(ReconstructionLossBase):

    def __init__(self, ssim_loss_weight=0.85, C1=1e-4, C2=9e-4, clip_loss=0.5, kernel_size=3, **kwargs):

        super().__init__(**kwargs)

        self.ssim_loss_weight = ssim_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.clip_loss = clip_loss
        self.kernel_size = kernel_size

    def photo_loss(self, source_image, target_image):
        ssim_value = SSIM(source_image, target_image, C1=self.C1, C2=self.C2, kernel_size=self.kernel_size)
        ssim_loss = torch.clamp((1. - ssim_value) / 2., 0., 1.)
        l1_loss = torch.abs(source_image - target_image)

        # average on the channel dimension
        if self.ssim_loss_weight > 0.0:
            photometric_loss = self.ssim_loss_weight * ssim_loss.mean(1, True) + \
                               (1 - self.ssim_loss_weight) * l1_loss.mean(1, True)
        else:
            photometric_loss = l1_loss

        if self.clip_loss > 0.0:
            mean, std = photometric_loss.mean(), photometric_loss.std()
            photometric_loss = torch.clamp(photometric_loss, max=float(mean + self.clip_loss * std))

        return photometric_loss



