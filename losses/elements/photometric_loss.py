import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.loss_base import LossBase


########################################################################################################################


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


########################################################################################################################

def compute_color_gradients(input_img, kernelx, kernely):

    input_img = (input_img* 255).floor()

    sobelx_r = F.conv2d(input_img[:, 0, ...].unsqueeze(1), kernelx, padding=1)
    sobelx_g = F.conv2d(input_img[:, 1, ...].unsqueeze(1), kernelx, padding=1)
    sobelx_b = F.conv2d(input_img[:, 2, ...].unsqueeze(1), kernelx, padding=1)

    sobely_r = F.conv2d(input_img[:, 0, ...].unsqueeze(1), kernely, padding=1)
    sobely_g = F.conv2d(input_img[:, 1, ...].unsqueeze(1), kernely, padding=1)
    sobely_b = F.conv2d(input_img[:, 2, ...].unsqueeze(1), kernely, padding=1)


    sobelx = torch.cat([sobelx_r, sobelx_g, sobelx_b], 1)
    sobely = torch.cat([sobely_r, sobely_g, sobely_b], 1)

    #grad_magnitude = torch.sqrt(sobelx.pow(2) + sobely.pow(2))

    # convert to uint8 [0,255]
    #grad_magnitude = grad_magnitude.byte()

    # convert to uint8 [0,255]
    #sobelx = sobelx.byte()
    #sobely = sobely.byte()

    return sobelx, sobely


def get_sobel_kernels():
    sobel_kernelx = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobel_kernelx = torch.Tensor(sobel_kernelx).expand(1, 1, 3, 3)

    sobel_kernely = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    sobel_kernely = torch.Tensor(sobel_kernely).expand(1, 1, 3, 3)

    return sobel_kernelx, sobel_kernely

########################################################################################################################

class PhotometricLoss(LossBase):
    """
    Self-Supervised photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them
    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scales to consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """

    def __init__(self, scales=4, ssim_loss_weight=0.85, C1=1e-4, C2=9e-4,
                 photometric_reduce_op='min', clip_loss=0.5, padding_mode='zeros', automask_loss=False):
        super().__init__()
        self.n = scales
        self.ssim_loss_weight = ssim_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss
        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter
        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales
        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i]) for i in range(self.n)]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3) for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss

        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

